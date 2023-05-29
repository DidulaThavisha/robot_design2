import torch
from utils.utils import AverageMeter,warmup_learning_rate,accuracy, save_model
import sys
import time
import numpy as np
from sklearn.metrics import precision_score,recall_score
from config.config_linear import parse_option
import torch.nn as nn
from loss.loss import add_kd_loss
from utils.utils import set_loader_new,set_model_student_teacher, set_model, set_optimizer, adjust_learning_rate
import pandas as pd
def main_student_teacher():
    best_acc = 0
    opt = parse_option()

    # build data loader
    device = opt.device
    train_loader,  test_loader = set_loader_new(opt)

    acc_list = []
    prec_list = []
    rec_list = []
    spec_list = []
    # training routine
    for i in range(0,3):
        model_teacher, model_student, criterion = set_model_student_teacher(opt)

        optimizer = set_optimizer(opt, model_student)
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train_OCT(train_loader, model_teacher, model_student, criterion,
                              optimizer, epoch, opt)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2 - time1, acc))

        loss, test_acc,prec,rec,spec = validate(test_loader, model_teacher, model_student, criterion, opt)
        acc_list.append(test_acc)
        prec_list.append(prec)
        rec_list.append(rec)
        spec_list.append(spec)
    df = pd.DataFrame({'Accuracy':acc_list,'Precision':prec_list,'Recall':rec_list,'Specificity':spec_list})
    excel_name = opt.backbone_training + '_' + opt.biomarker + '_' + str(opt.patient_split) + '.csv'
    df.to_csv(excel_name, index=False)

def train_OCT(train_loader, model_teacher, model_student, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model_teacher.eval()
    model_student.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    device = opt.device
    end = time.time()
    for idx, (image, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,eye_id,bcva,cst,patient) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = image.to(device)
        temperature = .1
        if (opt.biomarker == 'vit_deb'):
            labels = vit_deb
        elif (opt.biomarker == 'ir_hrf'):
            labels = ir_hrf
        elif (opt.biomarker == 'full_vit'):
            labels = full_vit
        elif (opt.biomarker == 'partial_vit'):
            labels = partial_vit
        elif (opt.biomarker == 'drt'):
            labels = drt
        else:
            labels = fluid_irf
        labels = labels.long()
        bsz = labels.shape[0]
        labels=labels.to(device)
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            output_teacher = model_teacher(images)

        output_student = model_student(images)

        loss = add_kd_loss(output_student, output_teacher,temperature)

        # update metric
        losses.update(loss.item(), bsz)

        acc1= accuracy(output_student, labels, topk=(1,))


        top1.update(acc1[0].item(), bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, model_student, criterion, opt):
    """validation"""
    model.eval()
    model_student.eval()
    device = opt.device
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    label_list = []
    out_list = []
    with torch.no_grad():
        end = time.time()
        for idx, (image, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,eye_id,bcva,cst,patient) in enumerate(val_loader):
            images = image.float().to(device)

            if (opt.biomarker == 'vit_deb'):
                labels = vit_deb
            elif (opt.biomarker == 'ir_hrf'):
                labels = ir_hrf
            elif (opt.biomarker == 'full_vit'):
                labels = full_vit
            elif (opt.biomarker == 'partial_vit'):
                labels = partial_vit
            elif (opt.biomarker == 'drt'):
                labels = drt
            else:
                labels = fluid_irf
            labels = labels.long()

            label_list.append(labels.detach().cpu().numpy())
            labels = labels.to(device)
            bsz = labels.shape[0]

            # forward
            output = model_student(images)

            loss = criterion(output, labels)
            _, pred = output.topk(1, 1, True, True)

            out_list.append(pred.detach().cpu().numpy())
            # update metri  c
            losses.update(loss.item(), bsz)
            acc1= accuracy(output, labels, topk=(1,))
            top1.update(acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    label_array = np.array(label_list)

    out_array = np.array(out_list)

    prec = precision_score(label_array.flatten(), out_array.flatten())
    rec = recall_score(label_array.flatten(), out_array.flatten())
    spec = recall_score(label_array.flatten(), out_array.flatten(), pos_label=0)


    return losses.avg, top1.avg,prec, rec, spec
