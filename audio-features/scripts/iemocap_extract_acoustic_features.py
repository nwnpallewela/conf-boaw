#!/bin/python
# python2
# Extract acoustic features (only llds) (csv)

import os

# Audio folders
# Folder containing the audio files of the test partition
folder_audio_train = "../data/iemo_trainx/"  # The audio files in the training partition converted to 16 bits are stored here
folder_audio_test = "../data/iemo_testx/"
# openSMILE
exe_opensmile = "..\\opensmile-3.0.2-windows-x86_64\\bin\\SMILExtract"  # MODIFY this path to the SMILExtract (version 2.3) executable
path_config = "..\\opensmile-3.0.2-windows-x86_64\\config\\compare16\\ComParE_2016.conf"  # MODIFY this path to the config file of openSMILE 2.3 - under Windows (cygwin): no POSIX

# Output files
outfilename_train = "..\\output\\iemo_75_audio_llds_train.csv"
outfilename_test = "..\\output\\iemo_75_audio_llds_test.csv"

# Clear LLD files
# if os.path.exists(outfilename_train):
#     os.remove(outfilename_train)
if os.path.exists(outfilename_test):
    os.remove(outfilename_test)

opensmile_options = "-configfile " + path_config + " -appendcsvlld 1 -timestampcsvlld 1 -headercsvlld 1"

# Extract features for train
for fn in os.listdir(folder_audio_train):
    infilename = folder_audio_train + fn
    instancename = os.path.splitext(fn)[0]
    outfilename = outfilename_train
    opensmile_call = exe_opensmile + " " + opensmile_options + " -inputfile " + infilename + " -lldcsvoutput " + outfilename + " -instname " + instancename
    os.system(opensmile_call)

# Extract features for test
for fn in os.listdir(folder_audio_test):
    infilename = folder_audio_test + fn
    instancename = os.path.splitext(fn)[0]
    outfilename = outfilename_test
    opensmile_call = exe_opensmile + " " + opensmile_options + " -inputfile " + infilename + " -lldcsvoutput " + outfilename + " -instname " + instancename
    os.system(opensmile_call)
