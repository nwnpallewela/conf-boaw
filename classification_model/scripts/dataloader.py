import pickle
import random

import arff
import numpy
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def adjust_label(param):
    if param == 0:
        return 0
    elif param == 1:
        return 1
    elif param == 2:
        return 2
    elif param == 3:
        return 3
    elif param == 4:
        return 4
    elif param == 5:
        return 5


class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True, fold=0):
        test_keys_array_fold_0 = ["Ses01F_impro02","Ses01F_impro05","Ses01F_script01_3","Ses01M_impro04","Ses01F_impro06","Ses02M_script01_3","Ses02F_impro03","Ses02F_script01_1","Ses02F_script03_1","Ses02F_impro04","Ses02F_impro02","Ses02F_script03_2","Ses03F_script03_1","Ses03F_impro07","Ses03F_script02_2","Ses03F_script01_2","Ses03F_impro05","Ses03F_script01_1","Ses03M_script01_2","Ses03M_script01_1","Ses03M_impro02","Ses04F_impro06","Ses04M_impro06","Ses04M_script02_2","Ses05M_script02_2","Ses05M_impro01","Ses05M_script01_3","Ses05F_impro05","Ses05F_impro03","Ses05M_impro08"]
        # test_keys_array_fold_0 = ['Ses02M_script02_2', 'Ses02M_script02_1', 'Ses01M_impro02', 'Ses01F_impro05',
        #                    'Ses03M_script03_2', 'Ses03F_script01_1', 'Ses02M_impro01', 'Ses03M_impro05a',
        #                    'Ses04F_script03_1', 'Ses05M_impro06', 'Ses05M_script01_2', 'Ses02M_impro08',
        #                    'Ses04M_script01_3', 'Ses04M_script01_2', 'Ses04F_script01_2', 'Ses03F_impro03',
        #                    'Ses01M_impro07', 'Ses03F_script02_2', 'Ses03F_impro08', 'Ses02F_script03_2',
        #                    'Ses03F_impro07', 'Ses02M_script01_2', 'Ses02M_impro03', 'Ses02M_script03_2',
        #                    'Ses02F_impro05', 'Ses04M_impro08', 'Ses05F_impro02', 'Ses05M_script03_2']

        test_keys_array_fold_1 = ['Ses01M_script02_2', 'Ses01F_script03_2', 'Ses01M_script01_1', 'Ses01F_impro02',
                                  'Ses01F_impro01',
                                  'Ses01F_impro05', 'Ses01M_impro06', 'Ses01F_script01_1', 'Ses01F_script03_1',
                                  'Ses01M_impro03',
                                  'Ses01F_script01_3', 'Ses01F_impro07', 'Ses01M_impro07', 'Ses01F_script02_1',
                                  'Ses01M_impro05',
                                  'Ses01M_script03_2', 'Ses01M_impro02', 'Ses01F_script02_2', 'Ses01M_impro04',
                                  'Ses01F_impro03',
                                  'Ses01M_script02_1', 'Ses01F_impro06', 'Ses01M_script03_1', 'Ses01M_script01_3',
                                  'Ses01F_impro04',
                                  'Ses01M_impro01', 'Ses01F_script01_2', 'Ses01M_script01_2']

        test_keys_array_fold_2 = ['Ses02M_impro06', 'Ses02M_impro04', 'Ses02M_script01_3', 'Ses02M_impro02',
                                  'Ses02M_script03_1',
                                  'Ses02F_impro07', 'Ses02M_impro03', 'Ses02M_impro08', 'Ses02F_impro03',
                                  'Ses02F_impro08', 'Ses02M_script01_1',
                                  'Ses02F_script01_1', 'Ses02F_impro05', 'Ses02F_script02_2', 'Ses02M_impro01',
                                  'Ses02F_script03_1',
                                  'Ses02F_impro01', 'Ses02M_impro05', 'Ses02F_script01_2', 'Ses02M_script03_2',
                                  'Ses02F_impro04',
                                  'Ses02F_impro06', 'Ses02F_script01_3', 'Ses02F_script02_1', 'Ses02F_impro02',
                                  'Ses02M_script02_1',
                                  'Ses02F_script03_2', 'Ses02M_script01_2', 'Ses02M_script02_2', 'Ses02M_impro07']

        test_keys_array_fold_3 = ['Ses03M_impro04', 'Ses03F_impro06', 'Ses03F_script03_1', 'Ses03M_script01_3',
                                  'Ses03M_script03_1', 'Ses03F_impro07'
            , 'Ses03M_impro03', 'Ses03M_script03_2', 'Ses03M_impro06', 'Ses03F_impro03', 'Ses03M_script02_1',
                                  'Ses03M_impro08a',
                                  'Ses03F_impro08', 'Ses03F_impro04', 'Ses03M_script02_2', 'Ses03F_script02_2',
                                  'Ses03F_script01_2', 'Ses03F_script01_3',
                                  'Ses03M_impro05b', 'Ses03F_impro05', 'Ses03F_impro02', 'Ses03M_impro05a',
                                  'Ses03F_script02_1',
                                  'Ses03F_script03_2', 'Ses03F_script01_1', 'Ses03M_impro07', 'Ses03M_script01_2',
                                  'Ses03F_impro01', 'Ses03M_script01_1', 'Ses03M_impro02', 'Ses03M_impro01',
                                  'Ses03M_impro08b']

        test_keys_array_fold_4 = ['Ses04F_impro08', 'Ses04F_script02_2', 'Ses04F_script01_3', 'Ses04M_impro07',
                                  'Ses04F_impro05',
                                  'Ses04M_script03_2', 'Ses04M_script01_3', 'Ses04M_impro08', 'Ses04M_impro04',
                                  'Ses04M_script03_1',
                                  'Ses04F_script03_1', 'Ses04M_impro02', 'Ses04F_impro02', 'Ses04F_script01_2',
                                  'Ses04F_impro01',
                                  'Ses04F_script02_1', 'Ses04M_impro01', 'Ses04F_script03_2', 'Ses04F_impro04',
                                  'Ses04F_impro03',
                                  'Ses04M_script02_1', 'Ses04F_impro06', 'Ses04M_impro06', 'Ses04M_script02_2',
                                  'Ses04F_script01_1',
                                  'Ses04F_impro07', 'Ses04M_script01_1', 'Ses04M_impro05', 'Ses04M_impro03',
                                  'Ses04M_script01_2']

        test_keys_array_fold_5 = ['Ses05M_impro06', 'Ses05F_script02_1', 'Ses05M_script01_1b', 'Ses05F_impro08',
                                  'Ses05M_impro03',
                                  'Ses05F_script01_3', 'Ses05F_impro01', 'Ses05M_script03_1', 'Ses05F_script01_2',
                                  'Ses05M_script01_1',
                                  'Ses05F_impro04', 'Ses05M_impro04', 'Ses05M_script02_2', 'Ses05F_script02_2',
                                  'Ses05F_script01_1',
                                  'Ses05M_impro02', 'Ses05M_impro01', 'Ses05F_script03_1', 'Ses05M_script01_3',
                                  'Ses05F_impro05',
                                  'Ses05M_script02_1', 'Ses05F_impro03', 'Ses05M_script03_2', 'Ses05M_impro05',
                                  'Ses05F_impro07',
                                  'Ses05M_impro08', 'Ses05M_script01_2', 'Ses05M_impro07', 'Ses05F_impro06',
                                  'Ses05F_script03_2',
                                  'Ses05F_impro02']
        fold_array = [test_keys_array_fold_0, test_keys_array_fold_1, test_keys_array_fold_2, test_keys_array_fold_3,
                      test_keys_array_fold_4, test_keys_array_fold_5]
        test_keys_array = fold_array[fold]
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        self.new_train_keys = []
        self.new_test_keys = []

        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = []
        if train:
            for x in self.trainVid:
                if x not in test_keys_array:
                    self.keys.append(x)
            for x in self.testVid:
                if x not in test_keys_array:
                    self.keys.append(x)
        else:
            self.keys = test_keys_array

        t_boaw = arff.load(open('../../audio-features/output/iemo_75_xbow_train.arff', 'r'))
        ts_boaw = arff.load(open('../../audio-features/output/iemo_75_xbow_test.arff', 'r'))
        self.train_boaw = np.array(t_boaw['data'])
        self.test_boaw = np.array(ts_boaw['data'])
        self.train_boaw_labels = pd.read_csv("../../audio-features/data/iemo_c_75_labels_train.csv", sep=';')
        self.test_boaw_labels = pd.read_csv("../../audio-features/data/iemo_c_75_labels_test.csv", sep=';')
        count = 0
        for row in self.train_boaw:
            row_label = self.train_boaw_labels.values[count][1]
            if row_label != row[2000]:
                print(self.train_boaw_labels.values[count][0])
            count = count + 1

        count = 0
        for row in self.test_boaw:
            row_label = self.test_boaw_labels.values[count][1]
            if row_label != row[2000]:
                print(self.test_boaw_labels.values[count][0])
            count = count + 1

        self.newVideoText = {}
        self.newVideoVisual = {}
        self.newVideoAudio = {}
        self.newVideoSpeakers = {}
        self.newVideoLabels = {}
        self.newVideoIds = {}

        merged_df = pd.concat([self.train_boaw_labels, pd.DataFrame(data=self.train_boaw)], axis=1)._append(
            pd.concat([self.test_boaw_labels, pd.DataFrame(data=self.test_boaw)], axis=1), ignore_index=True)

        self.codebook_boaw = {}
        transformed_df = merged_df
        removing_keys = set()
        for id in self.keys:
            self.codebook_boaw[id] = []
            self.newVideoText[id] = []
            self.newVideoVisual[id] = []
            self.newVideoAudio[id] = []
            self.newVideoSpeakers[id] = []
            self.newVideoLabels[id] = []
            self.newVideoIds[id] = []
            count = 0
            for v_key in self.videoIDs[id]:
                try:
                    # not_found = False
                    if v_key != 'Ses05M_script03_2_M044' and v_key != 'Ses05M_script03_2_M045':
                        values = np.array(transformed_df.loc[merged_df['id'] == v_key])[0][2:2002]
                        self.codebook_boaw[id].append(values.astype(float))
                        self.newVideoText[id].append(self.videoText[id][count])
                        self.newVideoVisual[id].append(self.videoVisual[id][count])
                        self.newVideoAudio[id].append(self.videoAudio[id][count])
                        self.newVideoSpeakers[id].append(self.videoSpeakers[id][count])
                        self.newVideoLabels[id].append(adjust_label(self.videoLabels[id][count]))
                        self.newVideoIds[id].append(v_key)
                except Exception as e:
                    print(id + " " + v_key)
                    removing_keys.add(id)
                finally:
                    count = count + 1
        for k in removing_keys:
            self.keys.remove(k)
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(numpy.array(self.newVideoText[vid])), \
               torch.FloatTensor(numpy.array(self.newVideoVisual[vid])), \
               torch.FloatTensor(numpy.array(self.codebook_boaw[vid])), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in \
                                  self.newVideoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.newVideoLabels[vid])), \
               torch.LongTensor(numpy.array(self.newVideoLabels[vid])), \
               self.newVideoIds[vid]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in
                dat]
