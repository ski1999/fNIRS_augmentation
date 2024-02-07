import os
import shutil
import time
import csv
import numpy as np
import pandas as pd
import random

def choose_aug(random_num,aug_num_org,aug_num):
  random.seed(random_num)
  aug_num_select = random.sample(list(range(aug_num_org)),aug_num)
  return aug_num_select

aug_size = 150
aug_num = 1
iteration = 20
timex
output_list_1 = np.empty((6000,100), dtype=int)
output_list_2 = np.empty((7500,100), dtype=int)
for fold in range(1,9):
  file_name = '~/CNN_BCI_code/Ensemble/Hamiltonian_output_' + str(fold) + '.csv'
  dist_list = pd.read_csv(file_name,header=None)
  dist_list.drop(columns=[0,1,2], axis=1, inplace=True)
  dist_list = dist_list.values
  if len(dist_list) == 6000:
    output_list_1 = np.append(output_list_1,dist_list,axis=0)
  elif len(dist_list) == 7500:
    output_list_2 = np.append(output_list_2,dist_list,axis=0)
output_list_1 = np.delete(output_list_1,0,axis=0)
output_list_2 = np.delete(output_list_2,0,axis=0)

ens_cnt_list = [] 
for itr in range(iteration):
  random_num = itr
  aug_num_select = choose_aug(random_num, aug_size, aug_num)
  ens_cnt = np.zeros((6,1,100), dtype=int)
  selected_list_1 = np.zeros((6,40,100),dtype=int)
  selected_list_2 = np.zeros((2,50,100),dtype=int)
  for t_b in aug_num_select:
    selected_list_1 = selected_list_1 + output_list_1[:,40*t_b:40*(t_b+1)-1,:]
    selected_list_2 = selected_list_2 + output_list_2[:,50*t_b:50*(t_b+1)-1,:]
  ens_cnt_1 = np.round(selected_list_1/aug_num,0)
  ens_cnt_2 = np.round(selected_list_2/aug_num,0)
  ens_pred_1 = np.sum(ens_cnt_1,axis=1)*2.5
  ens_pred_2 = np.sum(ens_cnt_2,axis=1)*2.0
  ens_predicted = np.append(ens_pred_1,ens_pred_2,axis=0)
  ens_pred_findmax = np.sum(ens_predicted,axis=0)
  ens_cnt_list.append(np.max(ens_pred_findmax))
np.mean(ens_cnt_list)
np.std(ens_cnt_list)

len_file = len_file/aug_size  
ensemble_label_predicted = []
    for t_a in range(len_file):
        ens_cnt = 0
        for t_b in range(aug_size): #should change if aug size change
            ens_cnt = ens_cnt + eval_indexFiles[(aug_size*t_a)+t_b][epoch+3]
        ens_cnt = round(ens_cnt/aug_size)
        ensemble_label_predicted.append(ens_cnt)
    ens_acc_list.append(sum(ensemble_label_predicted)*100/len_file)
    print('Ensemble Acc : %.3f %%\n' %(sum(ensemble_label_predicted)*100/len_file))