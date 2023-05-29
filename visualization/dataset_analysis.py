import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mayavi import mlab
import matplotlib

def trex_dme_concatenation(trex_dir):
    directories = os.listdir(trex_dir)
    total_data = []
    data = pd.DataFrame()
    for dir in directories:
        total_dir = os.path.join(trex_dir, dir)
        os.chdir(total_dir)
        excel = os.listdir(total_dir)

        for file in excel:
            df_excel_1 = pd.read_excel(file, 0)
            df_excel_2 = pd.read_excel(file, 1)
            # print(df_excel_2.head())
            data = data.append(df_excel_1)
            data = data.append(df_excel_2)
    os.chdir('/home/kiran/Desktop/Dev/SupCon')
    data.to_csv('./trex_dme/trex_dme_biomarkers.csv', index=False)


def trex_dme_analysis(pd_dir):
    df = pd.read_csv(pd_dir)
    for c in df.columns:
        print("---- %s ---" % c)
        print(df[c].value_counts())


def combine_trex_prime(trex_dir, prime_dir):
    trex_df = pd.read_csv(trex_dir)
    prime_df = pd.read_csv(prime_dir)
    trex_df = trex_df.append(prime_df)

    trex_df.to_csv('./trex_dme/trex_prime_combine_format_fixed.csv', index=False)


def trex_append_heading(trex_df):
    df = pd.read_csv(trex_df)
    for i in range(0, len(df)):
        path = df.iloc[i, 0]
        # path = os.path.join('data/Datasets/',path)
        path = '/' + path
        df.iloc[i, 0] = path
    df.to_csv('./trex_dme/trex_dme_biomarkers.csv', index=False)


def dataframe_extractor(data_dir):
    df = pd.DataFrame(columns=['File_Path', 'Patient_Code', 'Study', 'Week'])
    total_data = os.listdir(data_dir)
    os.chdir(data_dir)
    # GILA,Monthly,etc.
    for subset in total_data:
        folder_dir = os.path.join(data_dir, subset)
        subset_dir = os.listdir(folder_dir)
        os.chdir(folder_dir)
        # Which Patient
        for file in tqdm(subset_dir):
            patient_dir = os.path.join(folder_dir, file)
            patient_files = os.listdir(patient_dir)
            os.chdir(patient_dir)
            # Loop through weeks of each patient
            for week in patient_files:
                eye_dir = os.path.join(patient_dir, week)
                eye_files = os.listdir(eye_dir)
                os.chdir(eye_dir)
                # Loop through each patient's eye
                for eye in eye_files:
                    oct_dir = os.path.join(eye_dir, eye)
                    oct_files = os.listdir(oct_dir)
                    os.chdir(oct_dir)
                    for oct_file in oct_files:
                        im = Image.open(oct_file).convert("L")

                        width, height = im.size
                        if (width == height):
                            file_name = 'fundus_'+ str(eye) + '_' + str(week) + '.tif'
                            im.save(file_name)
                            #df.loc[len(df)] = [path, file, subset, week]

    #df.to_csv('/home/kiran/Desktop/Dev/SupCon/trex_dme/overall_trex_dme.csv', index=False)

    return 1


def trex_prime_comb_patient(df):
    df = pd.read_csv(df)
    df['Patient_ID'] = ""
    patient_num = 0
    patient_prev = ""
    patient_next = ""
    for i in range(0, len(df)):

        path = df.iloc[i, 0]
        split = path.split('/')
        if (i == 0):
            patient_num = 1
            patient_prev = split[5]
            patient_next = split[5]
        if (split[3] == 'TREX DME'):
            patient_next = split[5]
        else:
            patient_next = split[4]

        if (patient_next != patient_prev):
            patient_num = patient_num + 1
        df.iloc[i, 22] = patient_num
        if (split[3] == 'TREX DME'):
            patient_prev = split[5]
        else:
            patient_prev = split[4]
    df.to_csv('/home/kiran/Desktop/Dev/SupCon/trex_dme/trex_prime_combined_patient_format_fixed.csv', index=False)


def combined_train_test_split(df):
    df = pd.read_csv(df)
    df_trex = df[df['Patient_ID'] <= 56]
    df_trex_test = df_trex[df_trex['Patient_ID'] <= 10]
    df_prime = df[df['Patient_ID'] > 56]
    df_prime_test = df_prime[df_prime['Patient_ID'] <= 66]
    test = pd.concat([df_prime_test, df_trex_test])

    test.to_csv('/home/kiran/Desktop/Dev/SupCon/final_csvs_1/complete_biomarker_test.csv', index=False)


def file_format_conversion(df):
    df = pd.read_csv(df)
    for i in range(0, len(df)):
        loc = df.iloc[i, 0]
        split = loc.split('/')
        path = os.path.join('/', split[1], split[2], split[3], split[4], split[5], split[6], split[7])
        os.chdir(path)
        files = os.listdir()
        format_img = files[0][-4:]
        format_csv = split[8][-4:]
        if (format_img != format_csv):
            head = split[8][:-4]
            new_file = head + format_img
            df.iloc[i, 0] = os.path.join(path, new_file)

    df.to_csv('/home/kiran/Desktop/Dev/SupCon/trex_dme/trex_dme_biomarkers_format_fixed.csv', index=False)


def trex_attach_clinical(df):
    df = pd.read_csv(df)
    bcva_list = ['./trex_dme/excel_files/Monthly_BCVA.xlsx', './trex_dme/excel_files/Gila_BCVA.xlsx',
                 './trex_dme/excel_files/TREX_BCVA.xlsx']
    cst_list = ['./trex_dme/excel_files/Monthly_CST.xlsx', './trex_dme/excel_files/Gila_CST.xlsx',
                './trex_dme/excel_files/TREX_CST.xlsx']
    drss_list = ['./trex_dme/excel_files/Monthly_DRSS.xlsx', './trex_dme/excel_files/Gila_DRSS.xlsx',
                 './trex_dme/excel_files/TREX_DRSS.xlsx']
    df['BCVA'] = ""
    df['CST'] = ""
    df['DRSS'] = ""
    total_list = [bcva_list, cst_list, drss_list]
    for j in range(0, len(total_list)):
        list = total_list[j]
        for excel in list:
            df_attribute = pd.read_excel(excel)
            df_attribute.reset_index(drop=True, inplace=True)

            for i in tqdm(range(0, len(df))):
                patient = df.iloc[i, 1]
                week = df.iloc[i, 3]


                if (patient in df_attribute['Patient_Code'].unique()):
                    row = df_attribute.loc[df_attribute['Patient_Code'] == patient]


                    val = row.iloc[0][week]

                    if (j == 0):
                        df.iloc[i, 4] = val
                    elif (j == 1):
                        df.iloc[i, 5] = val
                    elif (j == 2):
                        df.iloc[i, 6] = val
    df.to_csv('/home/kiran/Desktop/Dev/SupCon/trex_dme/overall_trex_dme_eyes_dropped_attributes.csv', index=False)


def preprocess_eye_sides(df):
    df = pd.read_csv(df)
    drop_list = []
    for i in tqdm(range(0, len(df))):
        patient = df.iloc[i, 1]
        path = df.iloc[i, 0]
        path_list = path.split('/')

        eye_folder = path_list[7]

        patient_eye = patient[-2:]

        if (eye_folder != patient_eye):
            drop_list.append(i)

    df = df.drop(index=drop_list)
    df.to_csv('/home/kiran/Desktop/Dev/SupCon/trex_dme/overall_trex_dme_eyes_dropped.csv', index=False)


def patient_id_add(df_1, df_2):
    df_1 = pd.read_csv(df_1)
    df_2 = pd.read_csv(df_2)
    df_1['Patient_ID'] = ""
    for i in tqdm(range(0, len(df_1))):
        path = df_1.iloc[i, 0]
        path_list = path.split('/')
        patient_1 = path_list[4]
        for j in range(0, len(df_2)):
            path = df_2.iloc[j, 0]
            path_list = path.split('/')
            # print(path_list)
            pat_code_2 = path_list[4]
            if (patient_1 == pat_code_2):
                df_1.iloc[i, 1] = df_2.iloc[j, 27]
                break
    df_1.to_csv('/home/kiran/Desktop/Dev/SupCon/prime/prime_imageswithdata_patient_added.csv', index=False)


def remove_test_patients(df, list):
    df = pd.read_csv(df)
    df_test = df[df['Patient_ID'].isin(list)]
    # df_trex_test_2
    train = df.drop(df_test.index)
    train.to_csv('/home/kiran/Desktop/Dev/SupCon/final_csvs_3/full_prime_train.csv', index=False)
    df_test.to_csv('/home/kiran/Desktop/Dev/SupCon/final_csvs_3/full_prime_test.csv', index=False)


def append_attributes(df, df_2):
    df = pd.read_csv(df)
    df_2 = pd.read_csv(df_2)
    df['BCVA'] = ""
    df['CST'] = ""
    df['DRSS'] = ""
    for i in tqdm(range(0, len(df))):
        patient = df.iloc[i, 22]
        path = df.iloc[i, 0]


        row = df_2.loc[df_2['Path (Trial/Arm/Folder/Visit/Eye/Image Name)'] == path]

        if (row.empty):
            continue
        else:
            df.iloc[i, 23] = row.iloc[0, 23]
            df.iloc[i, 24] = row.iloc[0, 24]
            df.iloc[i, 25] = row.iloc[0, 25]
    df.to_csv('/home/kiran/Desktop/Dev/SupCon/trex_dme/train_fluirf.csv', index=False)


def remove_drss(df):
    df = pd.read_csv(df)
    df = df[df['DRSS'] >= 0]
    df.to_csv('/home/kiran/Desktop/Dev/SupCon/final_csvs_1/full_prime_train.csv', index=False)


def add_recovery(df):
    df = pd.read_csv(df)
    for i in range(0, len(df)):
        df.iloc[i, 1] = df.iloc[i, 1] + 100
    df.to_csv('/home/kiran/Desktop/Dev/SupCon/final_csvs_1/full_recovery_train.csv', index=False)


def check_nan(df):
    df = pd.read_csv(df)
    df = df['Vitreous debris']
    for i in range(0, len(df)):
        if (np.isnan(df[i])):
            print(i)


def test_biomarker_split(df):
    df = pd.read_csv(df)
    test_0 = df[df['DRT/ME'] == 0]
    test_1 = df[df['DRT/ME'] == 1]
    test_0 = test_0.sample(n=500, random_state=1)
    test_1 = test_1.sample(n=500, random_state=1)
    test = pd.concat([test_0, test_1])
    test.to_csv('/home/kiran/Desktop/Dev/SupCon/final_csvs_2/test_biomarker_sets/test_DRT_ME.csv', index=False)


def gen_compressed(df):
    df = pd.read_csv(df)
    df = df[["File_Path", "BCVA", 'DRSS', 'CST', 'Eye_ID', 'Patient_ID']]
    df.to_csv('./final_csvs_2/datasets_combined/prime_compressed.csv', index=False)


def discretize_labels(df_start, num_cuts):
    df = pd.read_csv(df_start)
    df['BCVA'] = pd.cut(df.BCVA, bins=num_cuts, labels=False)
    df['CST'] = pd.cut(df.CST, bins=num_cuts, labels=False)

    file_name = '/home/kiran/Desktop/Dev/SupCon/final_csvs_1/Discretized_Datasets/cuts_' + str(num_cuts) + '.csv'
    df.to_csv(file_name, index=False)
    print(df.head())


def unique_patient(df, path):
    df_name = df
    df = pd.read_csv(df)

    df['Patient_ID'] = ""
    for i in range(0, len(df)):
        path = df.iloc[i, 0]

        split = path.split('/')
        if (split[3] == 'TREX DME'):
            val = int(split[5][0:4])
            print(val)
            df.iloc[i, 4] = val
        else:
            df.iloc[i, 4] = df.iloc[i, 3]

    print(df.head())
    df.to_csv(df_name, index=False)


def prime_copy_eye(df):
    df = pd.read_csv(df)
    df['Patient_ID'] = ""
    for i in range(0, len(df)):
        df.iloc[i, 16] = df.iloc[i, 1]

    df.to_csv('/home/kiran/Desktop/Dev/SupCon/final_csvs_2/prime_imageswithdata_patient_added.csv', index=False)


def swap(df):
    df = pd.read_csv(df)
    for i in range(0, len(df)):
        path = df.iloc[i, 0]
        split = path.split('/')
        if (split[3] == 'Prime_FULL'):
            cst = df.iloc[i, 25]
            drss = df.iloc[i, 24]
            df.iloc[i, 25] = drss
            df.iloc[i, 24] = cst

    df.to_csv('/home/kiran/Desktop/Dev/SupCon/final_csvs_1/test_biomarker_sets/test_PAVF.csv', index=False)


def fill_fluirf(df, df_2):
    df = pd.read_csv(df)
    df_2 = pd.read_csv(df_2)
    for i in tqdm(range(0, len(df))):
        text = df.iloc[i, 0]
        for j in range(0, len(df_2)):
            text2 = df_2.iloc[j, 0]
            if (text == text2):
                bcva = df_2.iloc[j, 23]
                cst = df_2.iloc[j, 24]
                drss = df_2.iloc[j, 25]
                df.iloc[i, 23] = bcva
                df.iloc[i, 24] = cst
                df.iloc[i, 25] = drss
                break
    print(df.head())
    df.to_csv('/home/kiran/Desktop/Dev/SupCon/final_csvs_1/test_biomarker_sets/test_fluirf.csv', index=False)




def fix_dataframe(df):
    df_name = df
    df = pd.read_csv(df)
    df = df[["File_Path", "BCVA", 'Snellen', 'CST', 'Patient_ID']]

    df.to_csv(df_name, index=False)


def rm_drss(df):
    df_name = df
    df = pd.read_csv(df)

    df = df.drop(columns=['DRSS'])
    df.to_csv(df_name, index=False)


def normalize(arr):
    arr_min = np.min(arr)
    return (arr - arr_min) / (np.max(arr) - arr_min)


def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.viridis(c))

    plt.show()


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z


def process_volume(img_dir):
    path = img_dir
    img_dir = os.listdir(img_dir)
    ordered_img_list = []
    # Order Files in Correct Orientation of Volume
    for i in range(0, 49):
        for j in range(0, len(img_dir)):
            if (len(img_dir[j]) > 14):
                file_num = int(img_dir[j][6:12])
                if (i == file_num):
                    ordered_img_list.append(img_dir[j])
    # Construct 3D Object
    volume_list = []
    for img in ordered_img_list:
        sect = Image.open(os.path.join(path, img))
        sect = sect.resize((200, 200))
        sect = np.array(sect)
        volume_list.append(sect)
    vol_array = np.array(volume_list)
    vol_array = vol_array.T
    for i in range(5, 49, 10):
        mlab.volume_slice(vol_array, plane_orientation='z_axes', slice_index=i, colormap='gray')
    # mlab.volume_slice(vol_array, plane_orientation='z_axes', slice_index=34,colormap='gray')
    mlab.volume_slice(vol_array, plane_orientation='x_axes', slice_index=10, colormap='gray')
    mlab.volume_slice(vol_array, plane_orientation='y_axes', slice_index=10, colormap='gray')

    mlab.show()

    # mlab.savefig('test.png')


def clinical_assoc(df):
    df = pd.read_csv(df)
    cst_vector = []
    bcva_vector = []
    eye_vector = []
    for i in range(0, len(df)):
        cst = df.iloc[i, 2]
        bcva = df.iloc[i, 1]
        eye = df.iloc[i,3]
        if (cst not in cst_vector):
            cst_vector.append(cst)
        if (bcva not in bcva_vector):
            bcva_vector.append(bcva)
        if(eye not in eye_vector):
            eye_vector.append(eye)

    bcva_count_tracker = np.zeros((len(bcva_vector)))

    bcva_count_eye = np.zeros((len(bcva_vector)))

    cst_count_tracker = np.zeros((len(cst_vector)))
    cst_count_eye = np.zeros((len(cst_vector)))

    bcva_vector = np.array(bcva_vector)

    cst_vector = np.array(cst_vector)

    eye_count_tracker = np.zeros((len(eye_vector)))
    for j in tqdm(range(0, len(eye_vector))):
        eye_tracker = []
        target_eye = eye_vector[j]
        for i in range(0, len(df)):
            eye = df.iloc[i, 3]
            if (eye == target_eye):
                eye_count_tracker[j] += 1


    for j in tqdm(range(0, len(bcva_vector))):
        eye_tracker = []
        target_bcva = bcva_vector[j]
        for i in range(0, len(df)):
            bcva = df.iloc[i, 1]
            eye = df.iloc[i, 3]
            if (bcva == target_bcva):
                bcva_count_tracker[j] += 1
            if (bcva == target_bcva and (eye not in eye_tracker)):
                bcva_count_eye[j] += 1
                eye_tracker.append(eye)

    x = np.array(bcva_vector).T
    y = np.array(bcva_count_tracker).T
    x = x.astype(float)
    y = y.astype(float)

    for j in tqdm(range(0, len(cst_vector))):
        eye_tracker = []
        target_cst = cst_vector[j]
        for i in range(0, len(df)):
            cst = df.iloc[i, 2]
            eye = df.iloc[i, 3]
            if (cst == target_cst):
                cst_count_tracker[j] += 1
            if (cst == target_cst and (eye not in eye_tracker)):
                cst_count_eye[j] += 1
                eye_tracker.append(eye)
    
    #matplotlib.rcParams.update({'font.size': 45})

    plt.figure(1)
    plt.bar(cst_vector, cst_count_tracker,color='lime',edgecolor = 'black',linewidth= 0,alpha = .7)
    plt.xlabel('CST values')
    plt.ylabel('Number of Images')
    plt.grid(c='black')
    plt.title('Number of Images associated with each CST Value')
    plt.figure(2)
    plt.grid(c='black')
    
    plt.bar(bcva_vector, bcva_count_tracker,color='red',edgecolor = 'black',linewidth= 0,alpha = .7)
    plt.xlabel('BCVA values')
    plt.ylabel('Number of Images')
    plt.title('Number of Images associated with each BCVA Value')

    plt.figure(3)
    plt.bar(eye_vector,eye_count_tracker,color='aqua',edgecolor = 'black',linewidth= 0,alpha = .7)
    plt.xlabel('Eye Identifier Number')
    plt.ylabel('Number of Images')
    plt.grid(c='black')
    plt.title('Number of Images associated with each Eye')


    plt.show()

def dataframe_extractor_recovery(data_dir):
    df = pd.DataFrame(columns=['File_Path'])
    total_data = os.listdir(data_dir)
    os.chdir(data_dir)

    # patient
    for file in tqdm(total_data):
        patient_dir = os.path.join(data_dir, file)
        patient_files = os.listdir(patient_dir)
        os.chdir(patient_dir)
        # Loop through weeks of each patient
        for week in patient_files:
            eye_dir = os.path.join(patient_dir, week)
            eye_files = os.listdir(eye_dir)
            os.chdir(eye_dir)
            # Loop through each patient's eye
            for eye in eye_files:
                oct_dir = os.path.join(eye_dir, eye)
                oct_files = os.listdir(oct_dir)
                os.chdir(oct_dir)
                for oct_file in oct_files:
                    im = Image.open(oct_file).convert('L')

                    width, height = im.size
                    if (width == height):
                        file_name = 'fundus_' + str(eye) + '_' + str(week) + '.tif'
                        im.save(file_name)

                    #if(width == 504):
                        #img = np.array(im)
                        #img = img[0:496, 504:1008]
                        #image = Image.fromarray(img)
                        #image.save(oct_file)

                        #path = os.path.join(oct_dir, oct_file)


                        #df.loc[len(df)] = [path]

    #df.to_csv('/home/kiran/Desktop/Dev/SupCon_OCT_Clinical/clinical_datasets/recovery/overall_recovery_complete.csv', index=False)

    return 1

def percentage_training_set(csv_file,target_dir,frac):
    df = pd.read_csv(csv_file)
    df = df.sample(frac=frac, replace=False, random_state=1)
    df.to_csv(target_dir,index=False)

def biomarker_clinical_correlation(df_dir):
    df = pd.read_csv(df_dir)
    biomarker_list = [6,7,8,12,13]
    biomarker_name_list = ['IR HRF','PAVF','FAVF','DRT/ME','Fluid (IRF)']
    colors = ['blue','green','red','purple','orange']
    total_means = []
    total_std = []
    # Biomarker Not Present
    for j in biomarker_list:
        bcva_list_0 = []
        cst_list_0 = []
        bcva_list_1 = []
        cst_list_1 = []
        for i in range(0,len(df)):
        #DME
            if(df.iloc[i,j] == 0):
                bcva_list_0.append(df.iloc[i,23])
                cst_list_0.append(df.iloc[i, 24])
            else:
                bcva_list_1.append(df.iloc[i, 23])
                cst_list_1.append(df.iloc[i, 24])
        bcva_0 = np.array(bcva_list_0)
        bcva_1 = np.array(bcva_list_1)

        cst_0 = np.array(cst_list_0)
        cst_1 = np.array(cst_list_1)

        means = [np.mean(bcva_0), np.mean(cst_0),np.mean(bcva_1),np.mean(cst_1)]

        std = [np.std(bcva_0), np.std(cst_0),np.std(bcva_1),np.std(cst_1)]
        total_means.append(means)
        total_std.append(std)
    for k in range(0,len(total_means)):
        plt.scatter(total_means[k][0],total_means[k][1],c=colors[k],label = biomarker_name_list[k] + ' Absent',alpha=.2)
        plt.scatter(total_means[k][2], total_means[k][3],c=colors[k],label = biomarker_name_list[k] + ' Present',alpha=.99)
    plt.legend()
    plt.grid(c='black')
    plt.xlabel('Average BCVA Value')
    plt.ylabel('Average CST Value')
    plt.title('Clinical Values vs. Biomarker Presence')
    plt.show()

def excel_analysis(dir):
    excel_files = os.listdir(dir)
    names = ['CST','Eye ID','BCVA','SimCLR']
    # Average Positives and Negatives Plot

    pos_list = []
    neg_list = []
    plt.figure(1)
    for i in range(0,len(excel_files)):
        df = pd.read_excel(os.path.join(dir,excel_files[i]))
        pos_list.append(df.iloc[len(df)-1,1])
        neg_list.append(df.iloc[len(df)-1,2])
    x_axis = np.arange(len(names))
    plt.bar(x_axis - 0.2, pos_list, width=0.4, label='Positives')
    plt.bar(x_axis + 0.2, neg_list, width=0.4, label='Negatives')

    # Xticks

    plt.xticks(x_axis, names)
    plt.grid()
    # Add legend
    plt.xlabel('Strategy')
    plt.ylabel('Total')
    plt.title('Counts of Positives and Negatives per Strategy')
    plt.legend()

    # Display


    plt.figure(2)
    pos_list = []
    neg_list = []
    # Mean Positives and Negatives Plots
    for i in range(0,len(excel_files)):
        df = pd.read_excel(os.path.join(dir,excel_files[i]))
        pos_list.append(df.iloc[len(df) - 1, 3]*-1)
        neg_list.append(df.iloc[len(df) - 1, 4]*-1)
    x_axis = np.arange(len(names))
    plt.bar(x_axis - 0.2, pos_list, width=0.4, label='Positives')
    plt.bar(x_axis + 0.2, neg_list, width=0.4, label='Negatives')
    plt.xticks(x_axis, names)
    plt.grid()
    # Add legend
    plt.xlabel('Strategy')
    plt.ylabel('Average Mean Across Last Epoch')
    plt.title('Average Distance Statistics between positives and negatives')
    plt.legend()
    plt.show()

def discretize_clinical():
    pass

def res18_50_plots():
    pass

def biomarker_clin_percentages_plot():
    pass
if __name__ == '__main__':
    dir = '/home/kiran/Desktop/Dev/SupCon_OCT_Clinical/final_csvs_1/biomarker_csv_files/complete_biomarker_set.csv'
    biomarker_clinical_correlation(dir)