from utils import AverageMeter,warmup_learning_rate
import time
import torch
import sys
import numpy as np
import torch.nn as nn

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


class SupConLoss_SuperClassDistance(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, HCL=True, beta=1):
        super(SupConLoss_SuperClassDistance, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature
        self.beta = beta
        self.HCL = HCL

    def forward(self, features, labels=None, super_labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            super_labels.contiguous().view(-1, 1)
            super_labels = super_labels.reshape(len(super_labels), 1)
            super_mask = torch.eq(super_labels, super_labels.T).float().to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)

            super_labels.contiguous().view(-1, 1)
            super_labels = super_labels.reshape(len(super_labels), 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            mask = torch.eq(labels, labels.T).float().to(device)
            super_mask = torch.eq(super_labels, super_labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        super_mask = super_mask.repeat(anchor_count, contrast_count)
        super_logits = super_mask * logits
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        super_mask = super_mask * logits_mask
        neg_mask = logits_mask - super_mask

        # Super Negatives Only
        exp_logits_super = torch.exp(logits) * super_mask
        # Full Negatives without the SuperClass Negatives
        exp_logits = torch.exp(logits) * (neg_mask)
        # compute mean of log-likelihood over positive

        if (self.HCL == True):
            pos_matrix = torch.exp((mask * logits).sum())

            neg_log = torch.log(exp_logits_super.sum(1, keepdim=True))

            temperature = self.temperature

            tau_plus = .1
            N = batch_size * 2 - 2
            imp = (self.beta * neg_log).exp()
            reweight_neg = (imp * super_logits).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-tau_plus * N * pos_matrix + reweight_neg) / (1 - tau_plus)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
            log_prob = logits - Ng - torch.log(exp_logits.sum(1, keepdim=True))
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
def simclr_loss_func_hard(
    z: torch.Tensor,out_1:torch.Tensor,out_2:torch.Tensor, temperature: float = 0.5, beta:float=1, tau_plus:float=1) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """
    out = torch.cat([out_1, out_2], dim=0)

    device='cuda:0'
    batch_size = 128
    tau_plus = tau_plus
    beta = beta

    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    old_neg = neg.clone()
    mask = get_negative_mask(batch_size).to(device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    # negative samples similarity scoring

    N = batch_size * 2 - 2
    imp = (beta * neg.log()).exp()
    reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
    Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
        # constrain (optional)
    Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))


    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng))).mean()
    return loss
import torch.nn.functional as F
def train_Combined(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = opt.device
    end = time.time()
    for idx, (images, bcva,cst,eye_id,patient) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.to(device)

        bsz = bcva.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        #Method 1

        if opt.method1 == 'patient':
            labels1 = patient.cuda()
        elif opt.method1 == 'bcva':
            labels1 = bcva.cuda()
        elif opt.method1 == 'cst':
            labels1 = cst.cuda()

        elif opt.method1 == 'eye_id':
            labels1 = eye_id.cuda()
        else:
            labels1 = 'Null'
        # Method 2
        if opt.method2 == 'patient':
            labels2 = patient.cuda()
        elif opt.method2 == 'bcva':
            labels2 = bcva.cuda()
        elif opt.method2 == 'cst':
            labels2 = cst.cuda()

        elif opt.method2 == 'eye_id':
            labels2 = eye_id.cuda()
        else:
            labels2 = 'Null'
        # Method 3
        if opt.method3 == 'patient':
            labels3 = patient.cuda()
        elif opt.method3 == 'bcva':
            labels3 = bcva.cuda()
        elif opt.method3 == 'cst':
            labels3 = cst.cuda()

        elif opt.method3 == 'eye_id':
            labels3 = eye_id.cuda()
        else:
            labels3 = 'Null'
        # Method 4
        if opt.method4 == 'patient':
            labels4 = patient.cuda()
        elif opt.method4 == 'bcva':
            labels4 = bcva.cuda()
        elif opt.method4 == 'cst':
            labels4 = cst.cuda()

        elif opt.method4 == 'eye_id':
            labels4 = eye_id.cuda()
        else:
            labels4 = 'Null'
        # Method 5
        if opt.method5 == 'patient':
            labels5 = patient.cuda()
        elif opt.method5 == 'bcva':
            labels5 = bcva.cuda()
        elif opt.method5 == 'cst':
            labels5 = cst.cuda()

        elif opt.method5 == 'eye_id':
            labels5 = eye_id.cuda()
        else:
            labels5 = 'Null'

        if(opt.num_methods == 0):
            loss = criterion(features)
        elif(opt.num_methods==1):
            if (opt.method1 == 'HCL'):

                loss = simclr_loss_func_hard(features, f1, f2)
            else:
                loss= criterion(features,labels1)
        elif(opt.num_methods == 2):
            if(opt.method2 == 'SuperClass'):
                criterion = SupConLoss_SuperClassDistance(temperature=.07)
                loss = criterion(features,super_labels = labels1)
            elif(opt.method2 == 'SuperClass_Combined'):
                criterion2 = SupConLoss_SuperClassDistance(temperature=.07)
                loss = criterion(features) +  criterion2(features,super_labels = labels1)
            else:
                loss = criterion(features,labels1) + criterion(features,labels2)
        elif(opt.num_methods == 3):
            loss = criterion(features,labels1) + criterion(features,labels2) + criterion(features,labels3)
        elif (opt.num_methods == 4):
            loss = criterion(features, labels1) + criterion(features, labels2) + criterion(features, labels3) + criterion(features,labels4)
        elif (opt.num_methods == 5):
            loss = criterion(features, labels1) + criterion(features, labels2) + criterion(features, labels3) + criterion(features,labels4) + criterion(features,labels5)
        else:
            loss = 'Null'
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg
