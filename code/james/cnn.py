import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

import cv2
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d,avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

import sys

TRAIN_LABEL_DIR = 'datasets/train_labels.csv'
TRAIN_DIR_PRE = 'datasets/train_images_cropped.npy'
TEST_DIR_PRE = 'datasets/test_images_cropped.npy'

# parameter for image and classifer
NUM_CLASS = 32
IMG_SIZE = 35
LR = 1e-3

MODEL_NAME  = 'hand-drawn-{}-{}.model'.format(LR, '6conv-basic')

cate = ['sink','pear','moustache','nose','skateboard','penguin'
          ,'peanut','skull','panda','paintbrush','nail','apple',
          'rifle','mug','sailboat','pineapple','spoon','rabbit',
          'shovel','rollerskates','screwdriver','scorpion','rhinoceros'
          ,'pool','octagon','pillow','parrot','squiggle','mouth',
           'empty','pencil']

class HDI_Recognition():

    def category_maker(self):
        self.categories = []
        i=0
        for name in cate:
            self.categories.append([name,i])
            i+=1

    def preprocess(self):
        lb = preprocessing.LabelBinarizer()
        lb.fit(cate)
        #Notice here preprocessed image as input
        train_row = np.load(TRAIN_DIR_PRE, encoding = 'latin1')
        train_label = np.array(pd.read_csv(TRAIN_LABEL_DIR, delimiter=","))
        final_test = np.load(TEST_DIR_PRE,encoding='latin1')

        train_labels = train_label[:,1]
        train_x = train_row[:,1]
        train_data = []
        for i in range(len(train_labels)):
            encoded = lb.transform([train_labels[i]])
            train_data.append([np.array(train_x[i]), encoded[0]])

        train = train_data[:9000]
        valid = train_data[9000:9950]
        test = train_data[9950:]

        X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        Y = [i[1] for i in train]
        Y = np.reshape(Y,(-1,31))

        valid_x = np.array([i[0] for i in valid]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        valid_y = np.array([i[1] for i in valid]).reshape(-1,31)

        test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        test_y = np.array([i[1] for i in test]).reshape(-1,31)

        # image black is 0, dot is 39
        return X,Y,valid_x,valid_y,test_x,test_y
    """
This preprocess method does not binarize the lables, instead it returns lable with shape (7000,)

    """
    def preprocess_without_binarize(self):
        #This will give a 1-of-k coding scheme with alphbat sequence
        lb = preprocessing.LabelBinarizer()

        lb.fit(cate)
        lb.transform(['apple']).argmax(axis=-1)
        lb.transform(['apple','pear']).argmax(axis=-1)

        train_row = np.load(TRAIN_DIR_PRE, encoding = 'latin1')
        train_label = np.array(pd.read_csv(TRAIN_LABEL_DIR, delimiter=","))
        train_labels = train_label[:,1]
        train_x = train_row[:,1]
        train_data = []
        
        # print(train_labels[0:5])
        for i in range(len(train_labels)):
            encode = cate.index(train_labels[i])
            train_data.append([np.array(train_x[i]), encode])   

        train = train_data[:7000]
        valid = train_data[7000:9000]
        test = train_data[9000:]

        train_X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE)
        train_Y = np.array([i[1] for i in train]).reshape(len(train))
        # print(train_Y.shape)

        valid_x = np.array([i[0] for i in valid]).reshape(-1, IMG_SIZE, IMG_SIZE)
        valid_y = np.array([i[1] for i in valid]).reshape(len(valid))
        # print(valid_y.shape)

        test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE)
        test_y = np.array([i[1] for i in test]).reshape(len(test))
        # print(test_y.shape)

        return train_X,train_Y,valid_x,valid_y,test_x,test_y
    
    def cnn_tuning(self,X,Y,valid_x,valid_y,nbf1=16,fs1=2,nbf2=16,fs2=3,fc1=128,fc2=256,pool1='AVG',pool2='AVG',dropout_rate=0.2,batch_size=16,n_epoch=15,learning_rate=0.001):
        print("List of Hyperparameter for this training:\n")
        print("CNN l1 filter_numer / filter_size: " + str(nbf1) + ' | ' + str(fs1)+"\n")
        print("CNN l2 filter_numer / filter_size: " + str(nbf2) + ' | ' + str(fs2)+"\n")
        print("Pooling methods for pooling layer 1: " + str(pool1) + "\n")
        print("Pooling methods for pooling layer 2: " + str(pool2) + "\n")
        print("Dense layer neurons:  layer 1 | layer 2: " + str(fc1) + ' | ' + str(fc2)+"\n")
        print("Dropout rate: " + str(dropout_rate) +"\n")
        print("Learning_rate: " + str(learning_rate) +"\n")
        print("Batch_size: " + str(batch_size) +"\n")
        print("Epoch: " + str(n_epoch) +"\n")
        tf.reset_default_graph()
        convent = input_data(shape = [None, IMG_SIZE, IMG_SIZE, 1],name = 'input')

        # cnn 1 + pooling layer1
        convent = conv_2d(incoming = convent, 
                        nb_filter = nbf1, 
                        filter_size = fs1,
                        padding = 'valid',
                        activation = 'relu')
        if(pool1=='AVG'):
            convent = avg_pool_2d(incoming=convent, kernel_size=[2, 2], strides=[2, 2])
        else:
            convent = max_pool_2d(incoming=convent, kernel_size=[2, 2], strides=[2, 2])

        # cnn 1 + pooling layer1
        convent = conv_2d(incoming = convent, 
                        nb_filter = nbf2, 
                        filter_size = fs2,
                        padding = 'valid',
                        activation = 'relu')
        if(pool2=='AVG'):
            convent = avg_pool_2d(incoming=convent, kernel_size=[2, 2], strides=[2, 2])
        else:
            convent = max_pool_2d(incoming=convent, kernel_size=[2, 2], strides=[2, 2])

        convent = tf.contrib.layers.flatten(convent)
        convent = fully_connected(convent, fc1, activation = 'relu')
        convent = fully_connected(convent, fc2, activation = 'relu')
            #convent = fully_connected(convent, 512, activation = 'relu')
        convent = dropout(convent, dropout_rate)

        convent = fully_connected(convent, 31, activation ='softmax')
        convent = regression(convent, optimizer = 'Adam', learning_rate = learning_rate, loss='categorical_crossentropy', name = 'target')

        model = tflearn.DNN(convent, tensorboard_dir = 'log')

        model.fit(X, Y, n_epoch = n_epoch, validation_set = (valid_x,valid_y),
          batch_size = batch_size, snapshot_step=200, show_metric=True, run_id=MODEL_NAME)

    def show_sample_image(self,data,data_label,selection):
        # check preprocessing results
        plot = selection
        fig = plt.figure(figsize = (18,12))
        for i in range(len(plot)):
            img = data[plot[i]].reshape((IMG_SIZE,IMG_SIZE))
            label = data_label[plot[i]]
            subplot = fig.add_subplot(3,4,i+1)
            subplot.imshow(img, cmap ='gray_r')
            title=np.where(label==1)
            plt.title(title[0])
            subplot.axes.get_xaxis().set_visible(False)
            subplot.axes.get_yaxis().set_visible(False)
        plt.show()

if __name__ == '__main__':
    hdi = HDI_Recognition()
    hdi.category_maker()
    train_x,train_y,valid_x,valid_y,test_x,test_y = hdi.preprocess()
    #hdi.show_sample_image(train_x,train_y,[1,2,3,4])
    filters1 = [16,32,64,128] # number of filters for cnn1 
    filters2 = [16,32,64] # number of filters for cnn2
    filter_size_1 = [2,3,4,5]
    filter_size_2 = [2,3,4,5]
    fc_size_1 = [128,256,512,1024,2048]
    fc_size_2 = [64,128,256,512,1024]
    pooling = ['AVG','MAX']
    drop_out = [0.2,0.4,0.6,0.8]
    batch_size = [32,64,128,256]
    learning_rate = [0.001,0.002,0.004,0.008]
    # # batch size 
    # print("============================ Batch size ============================")
    # i = 1
    # for bs in batch_size:
    #     print("============================ Iteration: " + str(i) + "============================\n")
    #     hdi.cnn_tuning(train_x,train_y,valid_x,valid_y,batch_size=bs)
    #     i += 1 
    
    # drop out 
    # print("============================ Dropout rate ============================")
    # i = 1
    # for dr in drop_out:
    #     print("============================ Iteration: " + str(i) + "============================\n")
    #     hdi.cnn_tuning(train_x,train_y,valid_x,valid_y,dropout_rate=dr)
    #     i += 1

    # # filter size and number of filters
    # # print("============================ CNN Filter ============================")
    # i = 1
    # for f1 in filters1:
    #     for fs1 in filter_size_1:
    #         for f2 in filters2:
    #             for fs2 in filter_size_2:
    #                 print("============================ Iteration: " + str(i) + "============================\n")
    #                 hdi.cnn_tuning(train_x,train_y,valid_x,valid_y,nbf1=f1,fs1=fs1,nbf2=f2,fs2=fs2)
    #                 i += 1
    
    # neurons in dense layer 1 and 2
    # print("============================ Full Connected Layer ============================")
    # i = 1
    # for fc1 in fc_size_1:
    #     for fc2 in fc_size_2:
    #         print("============================ Iteration: " + str(i) + "============================\n")
    #         hdi.cnn_tuning(train_x,train_y,valid_x,valid_y,fc1=fc1,fc2=fc2)
    #         i += 1

    # pooling methods
    # print("============================ Pooling Layer ============================")
    # i = 1
    # for p1 in pooling:
    #     for p2 in pooling:
    #         print("============================ Iteration: " + str(i) + "============================\n")
    #         hdi.cnn_tuning(train_x,train_y,valid_x,valid_y,pool1=p1,pool2=p2)
    #         i += 1

    # # learning methods
    # print("============================ Leaarning Rate ============================")
    # i = 1
    # for lr in learning_rate:
    #         print("============================ Iteration: " + str(i) + "============================\n")
    #         hdi.cnn_tuning(train_x,train_y,valid_x,valid_y,learning_rate=lr)
    #         i += 1
    hdi.cnn_tuning(train_x,train_y,valid_x,valid_y,n_epoch=1000)
