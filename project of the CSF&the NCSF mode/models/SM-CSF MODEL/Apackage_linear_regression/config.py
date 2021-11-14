#import tensorflow as tf
import os
import glob
import json
import numpy as np
train_path="C:/Users/1/Desktop/paper_1/datas/combine/train/"
test_path="C:/Users/1/Desktop/paper_1/datas/combine/test/"

def get_input_dimension():
    #train_path_list=glob.glob(train_path+"/*")
    #for train in train_path_list:
    if glob.glob(train_path)==[]:
        print("data/train/xy_train is missing!!")
        exit()
    for line in open(train_path,"r",encoding="utf-8"):
        feature,label,original_feature=line.strip().split("\t")
        feature=json.loads(feature)
        return len(feature)
n_inp=get_input_dimension()
print(n_inp)
#n_inp=n_inp+1
stru_config={'n_inp':n_inp,'n_out':2}
#train_path="data/train"
#test_path="data/test/xy_test"
train_config={
        'CKPT':'ckpt',
        "new_train":True,
        "BATCHSIZE":10000,
        "MAX_ITER":5000,
        'STEP_SHOW':1500,
        'STEP_SAVE':1500,
        "LEARNING_RATE":0.005,
}
##########################################################

