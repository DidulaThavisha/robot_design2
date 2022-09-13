import pandas as pd

from config.config_linear import parse_option
import numpy as np
import torch

from tqdm import tqdm
from sklearn.manifold import TSNE

import matplotlib.patheffects as PathEffects
from utils.utils import TwoCropTransform
from models.resnet import  SupConResNet
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns

from datasets.biomarker import BiomarkerDatasetAttributes
from datasets.biomarker_severity import BiomarkerDatasetAttributes_Severity
import scipy.stats as stats

def set_model(opt):

    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()




    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    device = opt.device
    if torch.cuda.is_available():
        if opt.parallel == 0:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.to(device)

        criterion = criterion.to(device)
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model

def set_loader_new(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100' or opt.dataset == 'Ford' or opt.dataset == 'Ford_Region':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'OCT':
        mean = (.1904)
        std = (.2088)
    elif opt.dataset == 'Prime':
        mean = (.1706)
        std = (.2112)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform_SimCLR = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])




    data_path_train = opt.train_image_path
    csv_path_train = './final_csvs_' + str(opt.patient_split) +'/biomarker_csv_files/complete_biomarker_training.csv'
    if(opt.severity_analysis == 0):
        csv_path_test = './final_csvs_'+ str(opt.patient_split) +'/biomarker_csv_files/complete_biomarker_test.csv'
    else:
        csv_path_test = '/home/kiran/Desktop/Dev/gradcon-anomaly/test_set_grad_recon_loss.csv'


    data_path_test = opt.test_image_path


    train_dataset = BiomarkerDatasetAttributes(csv_path_train,data_path_train,transforms = TwoCropTransform(train_transform))
    if(opt.severity_analysis == 0):
        if(opt.model_type == 'SimCLR'):
            test_dataset = BiomarkerDatasetAttributes(csv_path_test,data_path_test,transforms = TwoCropTransform(val_transform_SimCLR))
        if(opt.noise_analysis == 1):
            test_dataset = BiomarkerDatasetAttributes(csv_path_test, data_path_test,transforms = TwoCropTransform(val_transform_SimCLR))
        else:
            test_dataset = BiomarkerDatasetAttributes(csv_path_test, data_path_test,val_transform)
    else:
        if (opt.noise_analysis == 1):
            test_dataset = BiomarkerDatasetAttributes_Severity(csv_path_test, data_path_test,transforms=TwoCropTransform(val_transform_SimCLR))
        else:
            test_dataset = BiomarkerDatasetAttributes_Severity(csv_path_test, data_path_test, val_transform)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    if(opt.biomarker == 'drt' and opt.patient_split == 1):
        dl = True
    elif(opt.multi == 1):
        dl = True
    else:
        dl=False
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True,drop_last=dl)

    return train_loader, test_loader
def analyze_mutual_information(loader,model,opt):
    df = pd.DataFrame(columns=["Vector", 'Labels'])
    with torch.no_grad():
        for idx, (images, vit_deb, ir_hrf, full_vit, partial_vit, fluid_irf, drt, eye_id, bcva, cst, patient) in enumerate(
                tqdm(loader)):
            images = torch.cat([images[0], images[1]], dim=0).to(opt.device)
            #features = model(images)
            #vec = features.squeeze().detach().cpu().numpy()
            df.loc[len(df)] = [images.detach().cpu().numpy(), bcva.detach().cpu().numpy()[0]]

            plt.imshow(df.iloc[0,0])
            plt.show()
def analyze_clinical(loader,model,opt):
    vector_list = []
    df = pd.DataFrame(columns=["Vector",'Labels'])

    if(opt.model_type !='SimCLR' and opt.noise_analysis == 0 and opt.severity_analysis == 0):
        with torch.no_grad():
            for idx, (images, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,eye_id,bcva,cst,patient) in enumerate(tqdm(loader)):
                images = images.to(opt.device)
                features = model(images)
                vec = features.squeeze().detach().cpu().numpy()
                df.loc[len(df)] = [vec, eye_id.detach().cpu().numpy()[0]]
    elif(opt.model_type !='SimCLR' and opt.noise_analysis == 1 and opt.severity_analysis == 0):
        with torch.no_grad():
            for idx, (images, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,eye_id,bcva,cst,patient) in enumerate(tqdm(loader)):
                images = torch.cat([images[0], images[1]], dim=0).to(opt.device)
                features = model(images)
                vec = features.squeeze().detach().cpu().numpy()
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
                df.loc[len(df)] = [vec[0], labels.detach().cpu().numpy()[0]]
                df.loc[len(df)] = [vec[1], labels.detach().cpu().numpy()[0]]
    elif (opt.model_type != 'SimCLR' and opt.noise_analysis == 1 and opt.severity_analysis == 1):
        with torch.no_grad():
            for idx, (images, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,eye_id,bcva,cst,gradcon_5000) in enumerate(tqdm(loader)):
                images = torch.cat([images[0], images[1]], dim=0).to(opt.device)
                features = model(images)
                vec = features.squeeze().detach().cpu().numpy()
                df.loc[len(df)] = [vec[0], gradcon_5000.detach().cpu().numpy()[0]]
                df.loc[len(df)] = [vec[1], gradcon_5000.detach().cpu().numpy()[0]]
    else:
        with torch.no_grad():
            for idx, (images, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,eye_id,bcva,cst,patient) in enumerate(tqdm(loader)):
                images = torch.cat([images[0], images[1]], dim=0).to(opt.device)
                features = model(images)
                vec = features.squeeze().detach().cpu().numpy()
                df.loc[len(df)] = [vec[0], drt.detach().cpu().numpy()[0]]
                df.loc[len(df)] = [vec[1], drt.detach().cpu().numpy()[0]]



    print(df['Labels'].unique())
    df.to_csv('./visualization/loss_analysis_csv/SimCLR_feature_vectors.csv',index=False)
def converter(instr):
    return np.fromstring(instr[1:-1],sep=' ')
def alignment_uniform_run(df_dir):
    df = pd.read_csv(df_dir,converters={'Vector':converter})
    unique_clinical_list = df['Labels'].unique()
    alignment_value_list = []
    uniformity_value_list = []
    for i in range(0,len(unique_clinical_list)):
        clinical_value_subset = df[df['Labels'] == unique_clinical_list[i]]
        alignment_value_list.append(alignment_calculation(clinical_value_subset))
        uniformity_value_list.append(uniformity_calculation(clinical_value_subset))

    with open(opt.results_dir, "a") as file:
        # Writing data to a file
        file.write(opt.ckpt + '\n')
        file.write(opt.biomarker + '\n')
        file.write('Alignment: ' + str(np.mean(np.array(alignment_value_list))) + '\n')
        file.write('Uniformity: ' + str(np.mean(np.array(uniformity_value_list))) + '\n')
def uniformity_calculation(df):
    total_values = []
    #Constructr Matrix
    for i in range(0, len(df)):
        if(i==0):
            pos_interest = df.iloc[i, 0].reshape((1, 128))
            pos_interest = torch.from_numpy(pos_interest)
            continue
        else:
            new_vector = df.iloc[i,0].reshape((1,128))
            new_vector = torch.from_numpy(new_vector)
            pos_interest = torch.cat([pos_interest,new_vector],dim=0)




    sq_pdist = torch.pdist(pos_interest,p=2).pow(2)


    return sq_pdist.mul(-2).exp().mean().log().numpy()
def alignment_calculation(df):
    total_values = []
    for i in range(0,len(df)):
        pos_interest = df.iloc[i,0].reshape((1,128))
        pos_interest = torch.from_numpy(pos_interest)
        value_array = []
        for j in range(0,len(df)):
            pos_compare = df.iloc[j,0].reshape((1,128))
            pos_compare = torch.from_numpy(pos_compare)
            if(j!=i):
                value_array.append((pos_interest-pos_compare).norm(dim=1).pow(2).mean().detach().numpy())

        total_values.append(np.mean(np.array(value_array)))

    return np.mean(np.array(total_values))
def main(opt):

    train_loader,test_loader = set_loader_new(opt)
    model = set_model(opt)
    analyze_mutual_information(test_loader,model,opt)
    #analyze_clinical(test_loader,model,opt)
    df = '/home/kiran/Desktop/Dev/SupCon_OCT_Clinical/visualization/loss_analysis_csv/SimCLR_feature_vectors.csv'
    #alignment_uniform_run(df)

if __name__ == '__main__':
    opt = parse_option()

    main(opt)