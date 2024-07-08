# %%
import pandas as pd
import numpy as np
import flowio
import warnings 
from tqdm import tqdm

def std(channel_data):
    # Calculate the mean
    mean = sum(channel_data) / len(channel_data)
    
    # Calculate the squared differences from the mean
    squared_diff = [(x - mean) ** 2 for x in channel_data]ju
    
    # Calculate the variance
    variance = sum(squared_diff) / len(channel_data)
    
    # Calculate the standard deviation (square root of the variance)
    std_dev = variance ** 0.5
    
    # Standardize the data
    STD = [(x - mean) / std_dev for x in channel_data]
    #STD = channel_data
    #STD = np.median(channel_data)
    return [STD]

def extract_features(patient_number):
    # given a patient number, open all relevant files and extract desired features. Return a single array of extracted features. 
    #NOTE: this is not the only way to set up your data matrix. Feel free to change this up. 

    feature_array = [patient_number] # start with the patient number in the first column so you can link labels. Remove prior to training/testing
    for sample in range(1,9):# 8 samples per patient      
        file_number = (patient_number-1)*8+sample # calculate the file number given the patient and sample numbers
        file_number_4d = '{:04d}'.format(file_number) # 4-digit file number (ex: 0007 instead of 7)

        # NOTE: CHANGE THIS TO THE CORRECT FILE PATH ON YOUR COMPUTER
        file_path = '/Users/gaberustia/Desktop/EGR410/Project_1_AML/FlowCytometry_files_1/' + file_number_4d + '.FCS'

        try:
            # Load the FCS file
            fcs_file_multi = flowio.read_multiple_data_sets(file_path,ignore_offset_error=True)
            # NOTE: there are 2 data sets in each file. The first is the real data, the second is control data (e.g. saline flush)
            # if you don't want to look at the second dataset, uncomment the following line: 
            del fcs_file_multi[-1] # this deletes the control portion of the file

            for ff in fcs_file_multi: 
                # Access metadata of the current data set
                metadata = ff.text
                # print("\nMetadata:", metadata)

                header = ff.header
                # print('\nHeader:', header)

                analysis = ff.analysis
                # print('\nAnalysis:', analysis)

                channel_count = ff.channel_count
                # print('\nChannel count:', channel_count)

                channels = ff.channels
                # print('\nChannels:', channels)

                event_count = ff.event_count
                # print('\nEvent count:', event_count)

                # Access data of the current data set
                data = ff.events
                # print('\nData length: ',len(data))

                for cc in range(0,channel_count):
                    channel_name = 'p'+str(cc+1)+'n'

                    channel_data = data[cc*event_count:(cc+1)*event_count-1]

                    my_feature = std(channel_data)

                    feature_array.append(my_feature) # assuming 'my_feature' is a single value, this will add 56 features (7 channels * 8 vials) for each patient 
                    # NOTE: if my_feature is an array instead of a single value, you will need to change up the structure on how you build your dataset for it to work properly. 

                    # It might be a good idea to look at some graphs...

                    # # Visualize channel distribution with a histogram 
                    # plt.hist(channel_data, bins=50, color='blue', alpha=0.7)
                    # plt.title(f'Histogram of {channel_name}')
                    # plt.xlabel('Fluorescence Intensity')
                    # plt.ylabel('Frequency')
                    # plt.show()

        except Exception as e:
            print('\n\nerror with file '+file_number_4d)
            print(e)

    return feature_array

warnings.simplefilter("ignore", category=UserWarning) # the .fcs file format throws a warning about data offsets. safe to ignore. alternative: 1 warning message every time you open a .fcs file (terminal now has 2500 warning messages)


# load patient numbers for training and testing 
training_patients_df = pd.read_csv('/Users/gaberustia/Desktop/EGR410/Project_1_AML/training_patients.csv')
testing_patients_df = pd.read_csv('/Users/gaberustia/Desktop/EGR410/Project_1_AML/testing_patients.csv')

training_set = [] # create placeholder
patients_train = training_patients_df['Patient Number'].to_list()
for patient in tqdm(patients_train, desc="Creating training set", unit="patients"): # 181 patients total
    training_set.append(extract_features(patient)) # create array of arrays (matrix)
df_train = pd.DataFrame(training_set) # NOTE: this doesn't have any column names. I suggest you keep track of which features are which. 
df_train.to_csv('training_set.csv',index=False)

testing_set = [] # create placeholder
patients_test = testing_patients_df['Patient Number'].to_list()
for patient in tqdm(patients_test, desc="Creating testing set", unit="patients"): # 178 patients total
    testing_set.append(extract_features(patient)) # create array of arrays (matrix)
df_test = pd.DataFrame(testing_set)
df_test.to_csv('testing_set.csv',index=False)


# %%
import statistics as st

def mean(channel_data):
    med = st.median(channel_data)
    return [med]

def extract_features2(patient_number):
    # given a patient number, open all relevant files and extract desired features. Return a single array of extracted features. 
    #NOTE: this is not the only way to set up your data matrix. Feel free to change this up. 

    feature_array = [patient_number] # start with the patient number in the first column so you can link labels. Remove prior to training/testing
    for sample in range(1,9):# 8 samples per patient      
        file_number = (patient_number-1)*8+sample # calculate the file number given the patient and sample numbers
        file_number_4d = '{:04d}'.format(file_number) # 4-digit file number (ex: 0007 instead of 7)

        # NOTE: CHANGE THIS TO THE CORRECT FILE PATH ON YOUR COMPUTER
        file_path = '/Users/gaberustia/Desktop/EGR410/Project_1_AML/FlowCytometry_files_1/' + file_number_4d + '.FCS'

        try:
            # Load the FCS file
            fcs_file_multi = flowio.read_multiple_data_sets(file_path,ignore_offset_error=True)
            # NOTE: there are 2 data sets in each file. The first is the real data, the second is control data (e.g. saline flush)
            # if you don't want to look at the second dataset, uncomment the following line: 
            del fcs_file_multi[-1] # this deletes the control portion of the file

            for ff in fcs_file_multi: 
                # Access metadata of the current data set
                metadata = ff.text
                # print("\nMetadata:", metadata)

                header = ff.header
                # print('\nHeader:', header)

                analysis = ff.analysis
                # print('\nAnalysis:', analysis)

                channel_count = ff.channel_count
                # print('\nChannel count:', channel_count)

                channels = ff.channels
                # print('\nChannels:', channels)

                event_count = ff.event_count
                # print('\nEvent count:', event_count)

                # Access data of the current data set
                data = ff.events
                # print('\nData length: ',len(data))

                for cc in range(0,channel_count):
                    channel_name = 'p'+str(cc+1)+'n'

                    channel_data = data[cc*event_count:(cc+1)*event_count-1]

                    my_feature = mean(channel_data)

                    feature_array.append(my_feature) # assuming 'my_feature' is a single value, this will add 56 features (7 channels * 8 vials) for each patient 
                    # NOTE: if my_feature is an array instead of a single value, you will need to change up the structure on how you build your dataset for it to work properly. 

                    # It might be a good idea to look at some graphs...

                    # # Visualize channel distribution with a histogram 
                    # plt.hist(channel_data, bins=50, color='blue', alpha=0.7)
                    # plt.title(f'Histogram of {channel_name}')
                    # plt.xlabel('Fluorescence Intensity')
                    # plt.ylabel('Frequency')
                    # plt.show()

        except Exception as e:
            print('\n\nerror with file '+file_number_4d)
            print(e)

    return feature_array

warnings.simplefilter("ignore", category=UserWarning) 

training_med = [] # create placeholder
patients_train = training_patients_df['Patient Number'].to_list()
for patient in tqdm(patients_train, desc="Creating training set", unit="patients"): # 181 patients total
    training_med.append(extract_features2(patient)) # create array of arrays (matrix)
df_train_med = pd.DataFrame(training_med) # NOTE: this doesn't have any column names. I suggest you keep track of which features are which. 


testing_med = [] # create placeholder
patients_test = testing_patients_df['Patient Number'].to_list()
for patient in tqdm(patients_test, desc="Creating testing set", unit="patients"): # 178 patients total
    testing_med.append(extract_features2(patient)) # create array of arrays (matrix)
df_test_med = pd.DataFrame(testing_med)


# %% [markdown]
# ## TRAINING

# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def gate2(cell_num, fsc, ssc, FSC_thres, SSC_thres):
    new_cell_num = []
    for i in range(cell_num):  # Iterate over indices instead of directly over cell_num
        if fsc[i] > FSC_thres and ssc[i] < SSC_thres:
            new_cell_num.append(i)  # Append index of the cell, not the cell itself
    return new_cell_num

def gate1(cell_num, cd45, ssc, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL):
    new_cell_num2 = []
    for i in range(cell_num):  # iterate over indices
        if cd45[i] > CD45_thresL and cd45[i] < CD45_thresH and ssc[i] < SSC_thresH and ssc[i] > SSC_thresL:
            new_cell_num2.append(i)
    return new_cell_num2



# %%
FSC_thres = -0.748
SSC_thres = 1000.5 
SSC_thresH = -0.4
SSC_thresL = -1.2 #-1.2
CD45_thresH = 1.75
CD45_thresL = 0.32

""" BEST 
F1: 2.5
CD45:  0.32:1.75
SSC: -1.2:-0.4 """

#rf = 0.8918918918918919
#svc = 0.9459459459459459
#dc = 0.918918918918919

#CD45: 0.3:1.7
#SS: -1.2:-0.4

# %%
df = pd.DataFrame(training_patients_df)

# Find cell positions where the Diagnosis is 'aml'
aml_positions = df.index[df['Diagnosis'] == 'aml'].tolist()

# Find cell positions where the Diagnosis is 'normal'
normal_positions = df.index[df['Diagnosis'] == 'normal'].tolist()

# Store the positions in an array
train_dia = {
    'aml': aml_positions,
    'normal': normal_positions
}

print(aml_positions)

# %%
num_of_cells = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_train)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_train.iloc[i, 1]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_of_cells.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_of_cells)

# %% [markdown]
# ## TRAINING GATES

# %%
gate_1 = []
for i in range(len(df_train)):
    fsc1 = np.array(df_train.iloc[i,1]).flatten()  # Remove [i] index
    ssc1 = np.array(df_train.iloc[i,2]).flatten()  # Remove [i] index
    cd451 = np.array(df_train.iloc[i,5]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_of_cells[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_1.append(gate10001)
#print(gate_1[1])

gate_2 = []
for i in range(len(df_train)):
    fsc2 = np.array(df_train.iloc[i,1]).flatten()
    ssc2 = np.array(df_train.iloc[i,2]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_1[i]]
    gate_ssc = [ssc2[loc] for loc in gate_1[i]]

    gate_num_cell = len(gate_1)

    gate20002 = gate2(len(gate_1[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_2.append(gate20002)

blast_ratio1 = []
for i in range(len(df_train)):
    ratio = len(gate_2[i])/(num_of_cells[i])
    blast_ratio1.append(ratio)

print(blast_ratio1)


# %%
num_of_cells2 = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_train)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_train.iloc[i, 9]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_of_cells2.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_of_cells2)

# %%
gate_01 = []
for i in range(len(df_train)):
    fsc1 = np.array(df_train.iloc[i,8]).flatten()  # Remove [i] index
    ssc1 = np.array(df_train.iloc[i,9]).flatten()  # Remove [i] index
    cd451 = np.array(df_train.iloc[i,12]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_of_cells2[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_01.append(gate10001)

gate_02 = []
for i in range(len(df_train)):
    fsc2 = np.array(df_train.iloc[i,8]).flatten()
    ssc2 = np.array(df_train.iloc[i,9]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_01[i]]
    gate_ssc = [ssc2[loc] for loc in gate_01[i]]

    gate_num_cell = len(gate_01)

    gate20002 = gate2(len(gate_01[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_02.append(gate20002)

blast_ratio2 = []
for i in range(len(df_train)):
    ratio = len(gate_02[i])/(num_of_cells2[i])
    blast_ratio2.append(ratio)

print(blast_ratio2)

# %%
num_of_cells3 = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_train)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_train.iloc[i, 15]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_of_cells3.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_of_cells3)

# %%
gate_001 = []
for i in range(len(df_train)):
    fsc1 = np.array(df_train.iloc[i,15]).flatten()  # Remove [i] index
    ssc1 = np.array(df_train.iloc[i,16]).flatten()  # Remove [i] index
    cd451 = np.array(df_train.iloc[i,19]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_of_cells3[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_001.append(gate10001)

gate_002 = []
for i in range(len(df_train)):
    fsc2 = np.array(df_train.iloc[i,15]).flatten()
    ssc2 = np.array(df_train.iloc[i,16]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_001[i]]
    gate_ssc = [ssc2[loc] for loc in gate_001[i]]

    gate_num_cell = len(gate_001)

    gate20002 = gate2(len(gate_001[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_002.append(gate20002)

blast_ratio3 = []
for i in range(len(df_train)):
    ratio = len(gate_002[i])/(num_of_cells3[i])
    blast_ratio3.append(ratio)

print(blast_ratio3)

# %%
num_of_cellsx = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_train)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_train.iloc[i, 22]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_of_cellsx.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_of_cellsx)

# %%
gate_x01 = []
for i in range(len(df_train)):
    fsc1 = np.array(df_train.iloc[i,22]).flatten()  # Remove [i] index
    ssc1 = np.array(df_train.iloc[i,23]).flatten()  # Remove [i] index
    cd451 = np.array(df_train.iloc[i,26]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_of_cellsx[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_x01.append(gate10001)

gate_x02 = []
for i in range(len(df_train)):
    fsc2 = np.array(df_train.iloc[i,22]).flatten()
    ssc2 = np.array(df_train.iloc[i,23]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_x01[i]]
    gate_ssc = [ssc2[loc] for loc in gate_x01[i]]

    gate_num_cell = len(gate_x01)

    gate20002 = gate2(len(gate_x01[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_x02.append(gate20002)

blast_ratiox3 = []
for i in range(len(df_train)):
    ratio = len(gate_x02[i])/(num_of_cellsx[i])
    blast_ratiox3.append(ratio)

print(blast_ratiox3)

# %%
num_of_cells4 = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_train)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_train.iloc[i, 29]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_of_cells4.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_of_cells4)

# %%
gate_101 = []
for i in range(len(df_train)):
    fsc1 = np.array(df_train.iloc[i,29]).flatten()  # Remove [i] index
    ssc1 = np.array(df_train.iloc[i,30]).flatten()  # Remove [i] index
    cd451 = np.array(df_train.iloc[i,33]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_of_cells4[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_101.append(gate10001)

gate_202 = []
for i in range(len(df_train)):
    fsc2 = np.array(df_train.iloc[i,29]).flatten()
    ssc2 = np.array(df_train.iloc[i,30]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_101[i]]
    gate_ssc = [ssc2[loc] for loc in gate_101[i]]

    gate_num_cell = len(gate_101)

    gate20002 = gate2(len(gate_101[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_202.append(gate20002)

blast_ratio4 = []
for i in range(len(df_train)):
    ratio = len(gate_202[i])/(num_of_cells4[i])
    blast_ratio4.append(ratio)

print(blast_ratio4)
# 5 = [33]
# 6 = [40]
# 8 = [54]

# %%
num_of_cells5 = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_train)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_train.iloc[i, 36]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_of_cells5.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_of_cells5)

# %%
gate_1001 = []
for i in range(len(df_train)):
    fsc1 = np.array(df_train.iloc[i,36]).flatten()  # Remove [i] index
    ssc1 = np.array(df_train.iloc[i,37]).flatten()  # Remove [i] index
    cd451 = np.array(df_train.iloc[i,40]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_of_cells5[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_1001.append(gate10001)

gate_2002 = []
for i in range(len(df_train)):
    fsc2 = np.array(df_train.iloc[i,36]).flatten()
    ssc2 = np.array(df_train.iloc[i,37]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_1001[i]]
    gate_ssc = [ssc2[loc] for loc in gate_1001[i]]

    gate_num_cell = len(gate_1001)

    gate20002 = gate2(len(gate_1001[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_2002.append(gate20002)

blast_ratio5 = []
for i in range(len(df_train)):
    ratio = len(gate_2002[i])/(num_of_cells5[i])
    blast_ratio5.append(ratio)

print(blast_ratio5)
# 5 = [33]
# 6 = [40]
# 8 = [54]

# %%
num_of_cellsx6 = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_train)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_train.iloc[i, 43]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_of_cellsx6.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_of_cellsx6)

# %%
gate_1xx1 = []
for i in range(len(df_train)):
    fsc1 = np.array(df_train.iloc[i,43]).flatten()  # Remove [i] index
    ssc1 = np.array(df_train.iloc[i,44]).flatten()  # Remove [i] index
    cd451 = np.array(df_train.iloc[i,47]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_of_cellsx6[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_1xx1.append(gate10001)

gate_2xx2 = []
for i in range(len(df_train)):
    fsc2 = np.array(df_train.iloc[i,43]).flatten()
    ssc2 = np.array(df_train.iloc[i,44]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_1xx1[i]]
    gate_ssc = [ssc2[loc] for loc in gate_1xx1[i]]

    gate_num_cell = len(gate_1xx1)

    gate20002 = gate2(len(gate_1xx1[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_2xx2.append(gate20002)

blast_ratiox5 = []
for i in range(len(df_train)):
    ratio = len(gate_2xx2[i])/(num_of_cellsx6[i])
    blast_ratiox5.append(ratio)

print(blast_ratiox5)


# %%
num_of_cells6 = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_train)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_train.iloc[i, 50]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_of_cells6.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_of_cells6)

# %%
gate_111 = []
for i in range(len(df_train)):
    fsc1 = np.array(df_train.iloc[i,50]).flatten()  # Remove [i] index
    ssc1 = np.array(df_train.iloc[i,51]).flatten()  # Remove [i] index
    cd451 = np.array(df_train.iloc[i,54]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_of_cells6[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_111.append(gate10001)

gate_222 = []
for i in range(len(df_train)):
    fsc2 = np.array(df_train.iloc[i,50]).flatten()
    ssc2 = np.array(df_train.iloc[i,51]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_111[i]]
    gate_ssc = [ssc2[loc] for loc in gate_111[i]]

    gate_num_cell = len(gate_111)

    gate20002 = gate2(len(gate_111[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_222.append(gate20002)

blast_ratio6 = []
for i in range(len(df_train)):
    ratio = len(gate_222[i])/(num_of_cells6[i])
    blast_ratio6.append(ratio)

print(blast_ratio6)
# 5 = [33]
# 6 = [40]
# 8 = [54]

# %%
FSC_channel = [1, 8, 15, 22, 29, 36, 43, 50]
SSC_channel = [2, 9, 16, 23, 30, 37, 44, 51]
FITC_channel = [3, 10, 17, 24, 31, 38, 45, 52]
PE_channel = [4, 11, 18, 25, 32, 39, 46, 53]
CD45_channel = [5, 12, 19, 26, 33, 40, 47, 54]
PC5_channel = [6, 13, 20, 27, 34, 41, 48, 55]
PC7_channel = [7, 14, 21, 28, 35, 42, 49, 56]

# %% [markdown]
# # OTHER TRAINING FEATURES

# %%
#print(training_patients_df)

training_results = training_patients_df.copy()

training_results['Diagnosis'] = training_results['Diagnosis'].map({'aml': 1, 'normal': 0})

#print(training_results)

# %%
def calculate_ratio(df_train_med, CD45, FITC):
    CD45_med2 = []
    FITC_med2 = []

    [CD45_med2.extend(sublist) for sublist in np.array(df_train_med.iloc[:, FSC_channel[CD45]])]
    [FITC_med2.extend(sublist) for sublist in np.array(df_train_med.iloc[:, SSC_channel[FITC]])]

    CD45_med2 = np.array(CD45_med2)
    FITC_med2 = np.array(FITC_med2)

    ratio2 = CD45_med2  #/ FITC_med2
    ratio2 = ratio2.tolist()
    
    return ratio2

ratio1 = calculate_ratio(df_train_med, 0, 0)
ratio2 = calculate_ratio(df_train_med, 1, 1)
ratio3 = calculate_ratio(df_train_med, 2, 2)
ratio4 = calculate_ratio(df_train_med, 3, 3)
ratio5 = calculate_ratio(df_train_med, 4, 4)
ratio6 = calculate_ratio(df_train_med, 5, 5)
ratio7 = calculate_ratio(df_train_med, 6, 6)
ratio8 = calculate_ratio(df_train_med, 7, 7)

# %%
def calculate_ratio2(df_train_med, CD45, channel):
    CD45_med2 = []

    [CD45_med2.extend(sublist) for sublist in np.array(df_train_med.iloc[:, channel[CD45]])]

    CD45_med2 = np.array(CD45_med2)

    ratio2 = CD45_med2  
    ratio2 = ratio2.tolist()
    
    return ratio2

num = 5

fsc_6 =calculate_ratio2(df_train_med, num, FSC_channel)
ssc_6 = calculate_ratio2(df_train_med, num, SSC_channel)
pe_6 = calculate_ratio2(df_train_med, num, PE_channel)
pc5_6 = calculate_ratio2(df_train_med, num, PC5_channel)
pc7_6 = calculate_ratio2(df_train_med, num, PC7_channel)
fitc_6 = calculate_ratio2(df_train_med, num, FITC_channel)

# %% [markdown]
# # MACHINE LEARNING

# %%
from sklearn import metrics
from sklearn.decomposition import PCA 

blast_data = { 'blast_ratio1': blast_ratio1,
            'blast_ratio2': blast_ratio2,
            'blast_ratio3': blast_ratiox3,
            'blast_ratio4': blast_ratio3,
            'blast_ratio5':blast_ratio4, 
            'blast_ratio6': blast_ratio5,
            'blast_ratio7': blast_ratiox5,
            'blast_ratio8':blast_ratio6, 

            'ratio1':ratio1, 
            'ratio2':ratio2,
            'ratio3':ratio3,
            'ratio4':ratio4,
            'ratio5':ratio5,
            'ratio6':ratio6,
            'ratio7':ratio7,
            'ratio8':ratio8, 

            'fsc_6': fsc_6,
            'ssc_6': ssc_6, 
            'pe_6': pe_6, 
            'pc5_6': pc5_6,
            'pc7_6': pc7_6,
            'fitc_6':fitc_6

            
            } # please ignore my stupid naming conventions

# use 1 3 7 
# 22 features 

# Convert dictionary to DataFrame
X = pd.DataFrame(blast_data)
y = training_results['Diagnosis']

pca_num = 10

pca = PCA(n_components=pca_num)
X2D = pca.fit_transform(X)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X2D, y, test_size=0.25, random_state=42,stratify=y) #

# 1 2 3 4 7

""" # Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) """
#  pca: 13


# %%
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

voting_clf = VotingClassifier(
        estimators=[
            #('lr', LogisticRegression(random_state=42)),
            #('rf', RandomForestClassifier(random_state=42)),
            #('svc', SVC(random_state=42)), 
            #('dc', DecisionTreeClassifier(random_state=42, max_depth=2,)),
            ('bd_rf', BaggingClassifier(RandomForestClassifier(),random_state=42)),
            ('bd_dc', BaggingClassifier(DecisionTreeClassifier(),random_state=42)),
            ('bd_et',BaggingClassifier(ExtraTreeClassifier(), random_state=42)),
] )

voting_clf.fit(X_train, y_train)

# %%
for name, clf in voting_clf.named_estimators_.items(): 
    print(name, "=", clf.score(X_test, y_test))

# %%
voting_clf.score(X_test, y_test)

# %%
def f1_score(y_true, y_pred):
    """
    Compute the F1 score.
    
    Parameters:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        
    Returns:
        float: F1 score.
    """
    # Calculate True Positives, False Positives, and False Negatives
    tp = sum((true == 1 and pred == 1) for true, pred in zip(y_true, y_pred))
    fp = sum((true == 0 and pred == 1) for true, pred in zip(y_true, y_pred))
    fn = sum((true == 1 and pred == 0) for true, pred in zip(y_true, y_pred))
    
    # Calculate Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1


# %%
voting_clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
y_vc = voting_clf.predict(X_test)


f1 = f1_score(y, y_vc)
print(f1)
print(voting_clf.score(X_test, y_test))

# %% [markdown]
# # TEST

# %%
#[3, 18, 29, 32, 44, 49, 55, 57, 66, 83, 89, 101, 107, 114, 118, 128, 134, 141, 142, 149, 155, 169, 173]


# %% [markdown]
# ## GATING FOR TEST

# %%
num_cells = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_test)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_test.iloc[i, 1]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_cells.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_cells)

# %%
gate_t1 = []
for i in range(len(df_test)):
    fsc1 = np.array(df_test.iloc[i,1]).flatten()  # Remove [i] index
    ssc1 = np.array(df_test.iloc[i,2]).flatten()  # Remove [i] index
    cd451 = np.array(df_test.iloc[i,5]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_cells[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_t1.append(gate10001)
#print(gate_1[1])

gate_t2 = []
for i in range(len(df_test)):
    fsc2 = np.array(df_test.iloc[i,1]).flatten()
    ssc2 = np.array(df_test.iloc[i,2]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_t1[i]]
    gate_ssc = [ssc2[loc] for loc in gate_t1[i]]

    gate_num_cell = len(gate_t1)

    gate20002 = gate2(len(gate_t1[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_t2.append(gate20002)

blast_t1 = []
for i in range(len(df_test)):
    ratio = len(gate_t2[i])/(num_of_cells[i])
    blast_t1.append(ratio)

print(blast_t1)

# %%
num_cells2 = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_test)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_test.iloc[i, 8]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_cells2.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_cells2)

# %%
gate_t01 = []
for i in range(len(df_test)):
    fsc1 = np.array(df_test.iloc[i,8]).flatten()  # Remove [i] index
    ssc1 = np.array(df_test.iloc[i,9]).flatten()  # Remove [i] index
    cd451 = np.array(df_test.iloc[i,12]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_cells2[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_t01.append(gate10001)
#print(gate_1[1])

gate_t02 = []
for i in range(len(df_test)):
    fsc2 = np.array(df_test.iloc[i,8]).flatten()
    ssc2 = np.array(df_test.iloc[i,9]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_t01[i]]
    gate_ssc = [ssc2[loc] for loc in gate_t01[i]]

    gate_num_cell = len(gate_t01)

    gate20002 = gate2(len(gate_t01[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_t02.append(gate20002)

blast_t2 = []
for i in range(len(df_test)):
    ratio = len(gate_t02[i])/(num_cells2[i])
    blast_t2.append(ratio)

print(blast_t2)

# %%
num_cells3 = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_test)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_test.iloc[i, 15]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_cells3.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_cells3)

# %%
gate_t001 = []
for i in range(len(df_test)):
    fsc1 = np.array(df_test.iloc[i,15]).flatten()  # Remove [i] index
    ssc1 = np.array(df_test.iloc[i,16]).flatten()  # Remove [i] index
    cd451 = np.array(df_test.iloc[i,19]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_cells3[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_t001.append(gate10001)
#print(gate_1[1])

gate_t002 = []
for i in range(len(df_test)):
    fsc2 = np.array(df_test.iloc[i,15]).flatten()
    ssc2 = np.array(df_test.iloc[i,16]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_t001[i]]
    gate_ssc = [ssc2[loc] for loc in gate_t001[i]]

    gate_num_cell = len(gate_t001)

    gate20002 = gate2(len(gate_t001[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_t002.append(gate20002)

blast_t3 = []
for i in range(len(df_test)):
    ratio = len(gate_t002[i])/(num_cells3[i])
    blast_t3.append(ratio)

print(blast_t3)

# %%
num_cells4 = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_test)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_test.iloc[i, 23]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_cells4.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_cells4)

# %%
gate_t0001 = []
for i in range(len(df_test)):
    fsc1 = np.array(df_test.iloc[i,22]).flatten()  # Remove [i] index
    ssc1 = np.array(df_test.iloc[i,23]).flatten()  # Remove [i] index
    cd451 = np.array(df_test.iloc[i,26]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_cells4[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_t0001.append(gate10001)
#print(gate_1[1])

gate_t0002 = []
for i in range(len(df_test)):
    fsc2 = np.array(df_test.iloc[i,22]).flatten()
    ssc2 = np.array(df_test.iloc[i,23]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_t0001[i]]
    gate_ssc = [ssc2[loc] for loc in gate_t0001[i]]

    gate_num_cell = len(gate_t0001)

    gate20002 = gate2(len(gate_t0001[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_t0002.append(gate20002)

blast_t4 = []
for i in range(len(df_test)):
    ratio = len(gate_t0002[i])/(num_cells4[i])
    blast_t4.append(ratio)

print(blast_t4)

# %%
num_cells5 = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_test)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_test.iloc[i, 29]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_cells5.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_cells5)

# %%
gate_t00001 = []
for i in range(len(df_test)):
    fsc1 = np.array(df_test.iloc[i,29]).flatten()  # Remove [i] index
    ssc1 = np.array(df_test.iloc[i,30]).flatten()  # Remove [i] index
    cd451 = np.array(df_test.iloc[i,33]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_cells5[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_t00001.append(gate10001)
#print(gate_1[1])

gate_t00002 = []
for i in range(len(df_test)):
    fsc2 = np.array(df_test.iloc[i,29]).flatten()
    ssc2 = np.array(df_test.iloc[i,30]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_t00001[i]]
    gate_ssc = [ssc2[loc] for loc in gate_t00001[i]]

    gate_num_cell = len(gate_t00001)

    gate20002 = gate2(len(gate_t00001[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_t00002.append(gate20002)

blast_t5 = []
for i in range(len(df_test)):
    ratio = len(gate_t00002[i])/(num_cells5[i])
    blast_t5.append(ratio)

print(blast_t5)

# %%
num_cells6 = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_test)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_test.iloc[i, 36]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_cells6.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_cells6)

# %%
gate_t00001 = []
for i in range(len(df_test)):
    fsc1 = np.array(df_test.iloc[i,36]).flatten()  # Remove [i] index
    ssc1 = np.array(df_test.iloc[i,37]).flatten()  # Remove [i] index
    cd451 = np.array(df_test.iloc[i,40]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_cells6[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_t00001.append(gate10001)
#print(gate_1[1])

gate_t00002 = []
for i in range(len(df_test)):
    fsc2 = np.array(df_test.iloc[i,36]).flatten()
    ssc2 = np.array(df_test.iloc[i,37]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_t00001[i]]
    gate_ssc = [ssc2[loc] for loc in gate_t00001[i]]

    gate_num_cell = len(gate_t00001)

    gate20002 = gate2(len(gate_t00001[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_t00002.append(gate20002)

blast_t6 = []
for i in range(len(df_test)):
    ratio = len(gate_t00002[i])/(num_cells6[i])
    blast_t6.append(ratio)

print(blast_t6)

# %%
num_cells7 = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_test)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_test.iloc[i, 43]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_cells7.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_cells7)

# %%
gate_t000001 = []
for i in range(len(df_test)):
    fsc1 = np.array(df_test.iloc[i,43]).flatten()  # Remove [i] index
    ssc1 = np.array(df_test.iloc[i,44]).flatten()  # Remove [i] index
    cd451 = np.array(df_test.iloc[i,47]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_cells7[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_t000001.append(gate10001)
#print(gate_1[1])

gate_t000002 = []
for i in range(len(df_test)):
    fsc2 = np.array(df_test.iloc[i,43]).flatten()
    ssc2 = np.array(df_test.iloc[i,44]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_t000001[i]]
    gate_ssc = [ssc2[loc] for loc in gate_t000001[i]]

    gate_num_cell = len(gate_t000001)

    gate20002 = gate2(len(gate_t000001[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_t000002.append(gate20002)

blast_t7 = []
for i in range(len(df_test)):
    ratio = len(gate_t000002[i])/(num_cells7[i])
    blast_t7.append(ratio)

print(blast_t7)

# %%
num_cells8 = []  # Initialize an empty list to store the number of cells for each patient

# Iterate over each row (patient) in df_train
for i in range(len(df_test)):
    # Get the number of cells for the current patient (assuming it's in the second column)
    cells_count = len(np.array(df_test.iloc[i, 50]).flatten())
    
    # Append the number of cells to the num_of_cells list
    num_cells8.append(cells_count)

# Now, num_of_cells contains the number of cells for each patient
    
print(num_cells8)

# %%
gate_t0000001 = []
for i in range(len(df_test)):
    fsc1 = np.array(df_test.iloc[i,50]).flatten()  # Remove [i] index
    ssc1 = np.array(df_test.iloc[i,51]).flatten()  # Remove [i] index
    cd451 = np.array(df_test.iloc[i,54]).flatten()  # Remove [i] index
    
    gate10001 = gate1(num_cells8[i], cd451, ssc1, SSC_thresH, SSC_thresL, CD45_thresH, CD45_thresL)  # Remove [i] index
    gate_t0000001.append(gate10001)
#print(gate_1[1])

gate_t0000002 = []
for i in range(len(df_test)):
    fsc2 = np.array(df_test.iloc[i,50]).flatten()
    ssc2 = np.array(df_test.iloc[i,51]).flatten()

    gate_fsc = [fsc2[loc] for loc in gate_t0000001[i]]
    gate_ssc = [ssc2[loc] for loc in gate_t0000001[i]]

    gate_num_cell = len(gate_t0000001)

    gate20002 = gate2(len(gate_t0000001[i]),gate_fsc,gate_ssc,FSC_thres,SSC_thres)
    gate_t0000002.append(gate20002)

blast_t8 = []
for i in range(len(df_test)):
    ratio = len(gate_t0000002[i])/(num_cells8[i])
    blast_t8.append(ratio)

print(blast_t8)

# %% [markdown]
# ## OTHER FEATURES FOR TEST

# %%
test_results = testing_patients_df.copy()

test_results['Diagnosis'] = test_results['Diagnosis'].map({'aml': 1, 'normal': 0})

# %%
ratio1_t = calculate_ratio(df_test_med, 0, 0)
ratio2_t = calculate_ratio(df_test_med, 1, 1)
ratio3_t = calculate_ratio(df_test_med, 2, 2)
ratio4_t = calculate_ratio(df_test_med, 3, 3)
ratio5_t = calculate_ratio(df_test_med, 4, 4)
ratio6_t = calculate_ratio(df_test_med, 5, 5)
ratio7_t = calculate_ratio(df_test_med, 6, 6)
ratio8_t = calculate_ratio(df_test_med, 7, 7)

# %%
num = 5

fsc_6t =calculate_ratio2(df_test_med, num, FSC_channel)
ssc_6t = calculate_ratio2(df_test_med, num, SSC_channel)
pe_6t = calculate_ratio2(df_test_med, num, PE_channel)
pc5_6t = calculate_ratio2(df_test_med, num, PC5_channel)
pc7_6t = calculate_ratio2(df_test_med, num, PC7_channel)
fitc_6t = calculate_ratio2(df_test_med, num, FITC_channel)


# %% [markdown]
# # MACHINE LEARNING FOR TEST

# %%
from sklearn import metrics
from sklearn.decomposition import PCA 

blast_test = {'blast_ratio1': blast_t1, 
              'blast_ratio2': blast_t2,
              'blast_ratio3': blast_t3, 
              'blast_ratio4': blast_t4,
              'blast_ratio5':blast_t5,
              'blast_ratio6': blast_t6,
              'blast_ratio7': blast_t7,
              'blast_ratio8':blast_t8,

              'ratio1_t':ratio1_t,
              'ratio2_t':ratio2_t,
              'ratio3_t':ratio3_t,
              'ratio4_t':ratio4_t,
              'ratio5_t':ratio5_t,
              'ratio6_t':ratio6_t,
              'ratio7_t':ratio7_t,
              'ratio8_t':ratio8_t,

              'fsc_6t':fsc_6t,
              'ssc_6t':ssc_6t,
              'pe_6t':pe_6t,
              'pc5_6t':pc5_6t,
              'pc7_6t':pc7_6t,
              'fitc_6t':fitc_6t

              } 

Xt = pd.DataFrame(blast_test)

pca = PCA(n_components= pca_num)
blast_pca = pca.fit_transform(Xt)

test_predict = voting_clf.predict(blast_pca)

def convert_binary_to_text(binary_array):
    text_array = []
    for bit in binary_array:
        if bit == 1:
            text_array.append("aml")
        elif bit == 0:
            text_array.append("normal")
        else:
            raise ValueError("Input array should only contain 0s and 1s")
    return text_array

text_predict = convert_binary_to_text(test_predict)
# print(text_predict)

# Assuming y_test_predict is a numpy array
# test_predict = [np.array(df_test.iloc[:, 0]).flatten().transpose()] +  [voting_clf.predict(blast_pca).transpose()]
# print(test_predict)

patient_nums = df_test.iloc[:,0]
diagnosis = text_predict # voting_clf.predict(text_predict).transpose()

# df_predict = pd.DataFrame(test_predict)
df_predict = pd.DataFrame({'Patient Number': patient_nums, 'Diagnosis': diagnosis})
df_predict.to_csv('Gabe_Rustia_predictions3.csv',index=False)

print(df_predict)

# Count the number of 1s
count_ones = np.count_nonzero(test_predict == 1)

print("Number of 1s in the array:", count_ones)

# %% [markdown]
# # Figures for Data

# %%
""" import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Sample data (assuming df_train, FSC_channel, and SSC_channel are defined)
pa1 = 169
pa2 = 4

# Assuming df_train, FSC_channel, and SSC_channel are defined

fig, axs = plt.subplots(2, 4, figsize=(16, 10))

# Overlay scatter plots for pa1 and pa2 for each subplot

# Loop through each subplot
for i in range(2):
    for j in range(4):
        # Scatter plot for vial (i*4 + j + 1)
        sns.scatterplot(x=np.array(df_train.iloc[pa1, CD45_channel[i*4 + j]]).flatten(),
                        y=np.array(df_train.iloc[pa1, SSC_channel[i*4 + j]]).flatten(),
                        ax=axs[i, j], color='blue', alpha=0.5, s=0.75)
        sns.scatterplot(x=np.array(df_train.iloc[pa2, CD45_channel[i*4 + j]]).flatten(),
                        y=np.array(df_train.iloc[pa2, SSC_channel[i*4 + j]]).flatten(),
                        ax=axs[i, j], color='red', alpha=0.5, s=0.75)
        axs[i, j].set_title('CD45 v SS Scatter plot vial {}'.format(i*4 + j + 1))

# Set x and y axis ticks
for ax in axs.flat:
    ax.set_xticks(np.arange(-1.5, 2.25, 0.5))
    ax.set_yticks(np.arange(-1.5, 2.25, 0.5))

plt.tight_layout()
plt.show()
 """

# %%
""" import seaborn as sns
import matplotlib.pyplot as plt

# Plot the first scatter plot


ra = ratio6
sns.scatterplot(x=df_train_med.iloc[:,0], y= ra, color='red', s=10)
sns.scatterplot(x=[df_train_med.iloc[i,0] for i in aml_positions], y=[ra[i] for i in aml_positions], color='blue', s=20)

plt.tight_layout()
plt.show()

# ratio: 6 """


