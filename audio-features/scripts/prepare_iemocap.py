#!/bin/python
# python2
# Prepare the files for emotion recognition with IEMOCAP
#  -Convert wav to 16 bits (openSMILE cannot read 32bit wave files)
#  -Split the data into a speaker-independent training (6 subjects: 11,12,13,14,15,16) and test partition (4 subjects: 03,08,09,10)
#  -Generate a labels file for both partitions
import pandas as pd
import os
import librosa
import soundfile as sf
# import noisereduce as nr

folder_audio_emodb = "../data/iemo_wav/"  # MODIFY this path to the folder with the iemocap audio files
folder_audio_train = "../data/iemo_trainx/"  # The audio files in the training partition converted to 16 bits are stored here
folder_audio_test  = "../data/iemo_testx/"   # The audio files in the test partition converted to 16 bits are stored here
temp = "../data/temp/"
labelsfilename_train = "../data/iemo_c_75_labels_train.csv"
# labelsfilename_train = "iemo_s_labels_train.csv"
labelsfilename_test  = "../data/iemo_c_75_labels_test.csv"
# labelsfilename_test  = "iemo_s_labels_test.csv"


confidence = pd.read_csv('../data/final_g_emotion_df.csv')
df = pd.read_csv('../data/iemocap_clip_labels/df_iemocap_1.csv')
df1 = pd.read_csv('../data/iemocap_clip_labels/df_iemocap_2.csv')
df2 = pd.read_csv('../data/iemocap_clip_labels/df_iemocap_3.csv')
df3 = pd.read_csv('../data/iemocap_clip_labels/df_iemocap_4.csv')
df4 = pd.read_csv('../data/iemocap_clip_labels/df_iemocap_5.csv')

final_df = pd.concat([df, df1, df2, df3, df4], ignore_index=True)
final_df.to_csv("../data/all_c_75_labels_from_iemocap.csv")
if not os.path.exists(folder_audio_train):
    os.mkdir(folder_audio_train)
if not os.path.exists(folder_audio_test):
    os.mkdir(folder_audio_test)

if os.path.exists(labelsfilename_train):
    os.remove(labelsfilename_train)
if os.path.exists(labelsfilename_test):
    os.remove(labelsfilename_test)

happy = 0
neu = 0
sad = 0
fru = 0
exe = 0
ang = 0
xxx = 0
thappy = 0
tneu = 0
tsad = 0
tfru = 0
texe = 0
tang = 0
txxx = 0

for fn in os.listdir(folder_audio_emodb):
    infilename   = folder_audio_emodb + fn
    instancename = os.path.splitext(fn)[0]
    
    label = final_df.loc[final_df['wav_file'] == instancename]['emotion'].values[0] # the label (target) is the 5th character in the filename
    # label = final_df.loc[final_df['wav_file'] == instancename]['val'].values[0] # the label (target) is the 5th character in the filename
    if label is None:
        # label = final_df.loc[final_df['wav_file'] == instancename]['val'].get(1)
        label = final_df.loc[final_df['wav_file'] == instancename]['emotion'].get(1)
    else:
        # if int(label) <= 2.5:
        #     label = 'n'
        # elif int(label) < 3.5:
        #     label = 'e'
        # else:
        #     label = 'p'
        max_val = confidence.loc[confidence['file'] == instancename].values[0][1:8].max()

        if label == 'fru':
            label = 'f'
        elif label == 'ang':
            label = 'a'
        elif label == 'xxx':
            label = 'x'
        elif label == 'hap':
            label = 'h'
        elif label == 'sad':
            label = 's'
        # elif label == 'fea':
        #     label = 'd'
        elif label == 'neu':
            label = 'n'
        elif label == 'exc':
            label = 'e'
        # elif label == 'sur':
        #     label = 'i'
        else:
            continue
    temp_file_name = temp + fn
    y, sr = sf.read(infilename)
    # y_reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(temp_file_name, y, sr)

    if (int(fn[3:5]) < 5) and max_val >= 0.75 and (label != 'x' or label != 's' or label != 'i' or label != 'd'):  # training partition

        outfilename = folder_audio_train + fn
        if label == 'f':
            fru = fru + 1
        elif label == 'a':
            ang = ang + 1
        elif label == 'h':
            happy = happy + 1
        elif label == 's':
            sad = sad + 1
        elif label == 'n':
            neu = neu + 1
        elif label == 'e':
            exe = exe + 1
        # elif label == 'x':
        #     xxx = xxx + 1
        with open(labelsfilename_train, 'a') as fl:
            fl.write(instancename + ';' + label + '\n')
    else:  # test partition
        outfilename = folder_audio_test + fn
        if label == 'f':
            tfru = tfru + 1
        elif label == 'a':
            tang = tang + 1
        elif label == 'h':
            thappy = thappy + 1
        elif label == 's':
            tsad = tsad + 1
        elif label == 'n':
            tneu = tneu + 1
        elif label == 'e':
            texe = texe + 1
        elif label == 'x':
            txxx = txxx + 1
        with open(labelsfilename_test, 'a') as fl:
            fl.write(instancename + ';' + label + '\n')

    sox_call = "sox " + temp_file_name + " -b 16 " + outfilename
    os.system(sox_call)



print("happy : " + str(happy))
print("sad : " + str(sad))
print("ang : " + str(ang))
print("fru : " + str(fru))
print("exe : " + str(exe))
print("neu : " + str(neu))
print("xxx : " + str(xxx))
print("test")
print("happy : " + str(thappy))
print("sad : " + str(tsad))
print("ang : " + str(tang))
print("fru : " + str(tfru))
print("exe : " + str(texe))
print("neu : " + str(tneu))
print("txxx : " + str(txxx))
