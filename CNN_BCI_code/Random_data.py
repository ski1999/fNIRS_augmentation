import os
import shutil
import time
import csv
import numpy as np
import pandas as pd
import random

def array_aug(data,given_list):
    new_test_list = [[0] * 52 for l in range(52)]
    for i in range(52):
        for k in range(52):
            #print(i, k)
            new_test_list[i][k] = data[given_list[i]][given_list[k]]
    return new_test_list

def shuffle_aug(data,itr):
    new_test_list = data.sample(frac=1, random_state=itr).reset_index(drop=True)
    return new_test_list

def function_shuffle_aug(itr,hem,path=None):
    if hem == 'n':
        func_arr = ((1,20),(2,3,12,13,23,8,9,18,19,30),(4,5,6,7,15,16,17,25,26,27,28,35,36,37),(10,),(11,22,33,21,31,41),(14,24,34,45,29,39,40,50),(32,42),(38,46,47,48,49),(43,44,51,52))
    elif hem == 'y':
        func_arr = ((1,),(20,),(2,3,12,13,23),(8,9,18,19,30),(4,5,6,7,15,16,17,25,26,27,28,35,36,37),(10,),(11,22,33),(21,31,41),(14,24,34,45),(29,39,40,50),(32,),(42,),(38,46,47,48,49),(43,44),(51,52))
    random.seed(itr % 10)
    new_func_list = []
    if path is None:
        pos_rand = list(range(len(func_arr)))
        random.shuffle(pos_rand)
        for i in range(len(func_arr)):
            inner = func_arr[pos_rand[i]]
            inner_rand = list(inner)
            random.seed(itr)
            random.shuffle(inner_rand)
            new_func_list.extend(inner_rand)
        new_func_list = [new_func_list[i] - 1 for i in range(len(new_func_list))]
        return new_func_list
    else:
        for i in range(len(func_arr)):
            inner = func_arr[path[i]]
            inner_rand = list(inner)
            random.seed(itr)
            random.shuffle(inner_rand)
            new_func_list.extend(inner_rand)
        new_func_list = [new_func_list[i] - 1 for i in range(len(new_func_list))]
        return new_func_list


def channel_hamiltonian():
    SupGR = [(12,2,23,13,3),(12,13,23,2,3),(12,13,3,2,23),(12,23,2,3,13),(12,23,13,2,3),(12,23,2,13,3),(2,3,13,12,23),(2,23,12,13,3),()]

#[PSC_R,PSC_L,SupG_R,SupG_L,SAC,SA,STG_R,STG_L,Ang_R,Ang_L,MTG_R,MTG_L,V3,FusG_R,FusG_L]
dist_list = pd.read_csv("~/CNN_BCI_code/Hamiltonian_output.csv",header=None, nrows = 2400)
dist_list = dist_list.values.tolist()
dist_list = dist_list[0::4][:]
region_list_y = pd.read_csv("~/CNN_BCI_code/Hamiltonian_output_region_hem_y.csv",header=None)
region_list_n = pd.read_csv("~/CNN_BCI_code/Hamiltonian_output_region_hem_n.csv",header=None)
region_list_y = region_list_y.values.tolist()
region_list_n = region_list_n.values.tolist()
#print(*dist_list, sep='\n')
#print(len(dist_list))

"""
random_list = list(range(52))
random_given = []
#random list
for rand_cnt in range(100):
    random.shuffle(random_list)
    random_given.append(random_list)
    random_list = list(range(52))
with open('random_list.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(random_given)
"""

path = "/home/ski1999/CNN_pcc_noaug_fold"
mode = "Hamiltonian" #"Random", "Functional_Random"
aug_size = 30
hem_flag = 'n' #Only for mode "Functional_Random"
new_fold_name = "CNN_pcc_funcv5_fold" #fold name
directory = os.listdir(path)
os.chdir(path)

#dist_list = [0,10,31,21,42,43,22,1,12,11,32,33,34,13,2,23,44,45,24,3,4,5,26,15,14,25,46,35,36,37,16,6,7,28,17,18,39,40,29,8,9,19,20,41,30,51,50,49,38,27,48,47]

#Hamiltonian path
if mode == "Hamiltonian":
    for (path,dir,files) in os.walk(path):
        for file in files:
            new_path = os.path.join(path,file)
            fileMatrix = []
            datafile = []
            with open(new_path,'r') as open_file:
                read_file = csv.reader(open_file)
                for line in read_file:
                    datafile.append(line)
                #print(len(datafile))
            for cnt in range(aug_size):
                aug_data = array_aug(datafile,dist_list[cnt])
                new_filename = 'aug_'+str(cnt+1)+'_'+file
                new_filename = os.path.join(path,new_filename)
                new_filename_dir = new_filename.split("/")
                new_filename_dir[3] = new_fold_name
                new_filename = "/".join(new_filename_dir)
                #print(new_filename)
                with open(new_filename,'w',newline='') as f:
                    write = csv.writer(f)
                    write.writerows(aug_data)


#Random Path
elif mode = "Random":
    for (path,dir,files) in os.walk(path):
        for file in files:
            new_path = os.path.join(path,file)
            datafile = pd.read_csv(new_path,header=None)
            for cnt in range(aug_size):
                aug_data = shuffle_aug(datafile,cnt)
                new_filename = 'aug_'+str(cnt+1)+'_'+file
                new_filename = os.path.join(path,new_filename)
                new_filename_dir = new_filename.split("/")
                new_filename_dir[3] = new_fold_name
                new_filename = "/".join(new_filename_dir)
                #print(new_filename)
                aug_data.to_csv(new_filename, index=False)

#Functional Random Path (Region ordering Hamiltonian)
elif mode = "Functional_Random":
    if hem_flag == 'y':
        region_list = region_list_y
    elif hem_flag == 'n':
        region_list = region_list_n
    for (path,dir,files) in os.walk(path):
        for file in files:
            new_path = os.path.join(path,file)
            datafile = pd.read_csv(new_path,header=None)
            for cnt in range(150):
                random.seed(cnt)
                select_path = random.randint(0,len(region_list)-1)
                #region_path = region_list[select_path][:]
                #functional_list = function_shuffle_aug(cnt,hem_flag,region_path)
                functional_list = function_shuffle_aug(cnt,hem_flag)
                #print(len(functional_list))
                #print(new_path)
                aug_data = array_aug(datafile, functional_list)
                new_filename = 'aug_'+str(cnt+1)+'_'+file
                new_filename = os.path.join(path,new_filename)
                new_filename_dir = new_filename.split("/")
                new_filename_dir[3] = new_fold_name
                new_filename = "/".join(new_filename_dir)
                #print(new_filename)
                with open(new_filename,'w',newline='') as f:
                    write = csv.writer(f)
                    write.writerows(aug_data)
