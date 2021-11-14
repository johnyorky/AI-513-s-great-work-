#import tensorflow as tf
import os
import glob
import json
import numpy as np

train_path="C:/Users/1/Desktop/paper_1/datas/combine/train"
data_test="C:/Users/1/Desktop/paper_1/datas/combine/test/xy_test_combine"


def get_input_dimension():
    train_path_list=glob.glob(train_path+"/*")
    for train in train_path_list:
        if glob.glob(train)==[]:
            print("xy_train is missing!!")
            exit()
        for line in open(train,"r",encoding="utf-8"):
            feature,label=line.strip().split("\t")
            feature=feature.split("<=>")#(feature)
            return len(feature)

n_inp=get_input_dimension()
stru_config={'n_inp':n_inp,
             'n_out':2,
             "which_model":"LSTM",#CNN
             "embedding":True,
             "embedding_size":16,
             "drop_out":0.5,
             "num_size":200,
             "full_connect_size":64}

train_config={
        'CKPT':'ckpt',
        "new_train":True,
        "BATCHSIZE":512,
        "MAX_ITER":5000,
        'STEP_SHOW':1000,
        'STEP_SAVE':1000,
        "LEARNING_RATE":0.005,
}
##########################################################

