import numpy as np
import pandas as pd

filedir = './'
filename = {
        'best': '%s.npz',
        'model': 'model_%s_r600800e%s.h5',
        'weight': 'model_weights_%s_r600800e%s.h5',
        'out': '%s.csv',
        }
disease_list = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia',
        ]





import os
from PIL import Image # pip install pillow
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import Sequence
from keras.models import Sequential,Model,load_model
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers import Dense,GaussianNoise,concatenate,Input,BatchNormalization,InputLayer
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD,Adam
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
#from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing import image
from keras.layers.advanced_activations import *
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import scale
from keras.applications.densenet import DenseNet121
from keras import backend as K
from keras.layers import Add
from keras.layers import LeakyReLU
from keras.regularizers import l1_l2
from keras.constraints import min_max_norm
from sklearn.metrics import log_loss
from densenet121 import densenet121_model
import skimage.measure
from custom_layers.scale_layer import Scale
from keras.layers import ZeroPadding2D
from keras.layers.core import Layer
from keras.engine import InputSpec
from keras.layers.advanced_activations import *
import cv2 #conda install opencv
from keras.layers import merge
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
from sklearn.metrics import roc_auc_score
from time import time



disease={'Atelectasis':0,
 'Cardiomegaly':1,
 'Effusion':2,
 'Infiltration':3,
 'Mass':4,
 'Nodule':5,
 'Pneumonia':6, 
 'Pneumothorax':7,
 'Consolidation':8, 
 'Edema':9,
 'Emphysema':10,
 'Fibrosis':11,
 'Pleural_Thickening':12, 
 'Hernia':13}

train_data_type={"all_data":0}

img_para={'dim1':600,
          'dim2':800}

path= {'img':'../../images/',
       'train':'train2.csv',
       'test':'test.csv',
       'sample':'sample_submission.csv',
       'all_label':"all_label3.csv",
#        'ans1':"300400e1.csv",
#        'ans2':"300400e2.csv",
#        'ans3':"300400e3.csv",
#        'ans4':"300400e4.csv",
#        'ans5':"300400e5.csv"
       'ans1':"600800e1.csv",
       'ans2':"600800e2.csv",
       'ans3':"600800e3.csv",
       'ans4':"600800e4.csv",
       'ans5':"600800e5.csv",
      }

para={'dropout':0.0,
      "weight_decay":10e-4,
      'epochs':1,
#       'round1':'300400e1',## format 8(FYI)+100(dim1)+134(dim2)+epoch(number)
#       'round2':'300400e2',
#       'round3':'300400e3',
#       'round4':'300400e4',
#       'round5':'300400e5',
      'round1':'600800e1',## format 8(FYI)+100(dim1)+134(dim2)+epoch(number)
      'round2':'600800e2',
      'round3':'600800e3',
      'round4':'600800e4',
      'round5':'600800e5',
      "batch":2, #120,100,9,2,1
      "val_seed":33652,
      'train_seed':30000}##limit is len(TesIdList)=33652
target=['NTUML']



# AUC for a binary classifier
def auc(y_true, y_pred):
    #y_true= tf.convert_to_tensor(y_true)
    y_true = tf.cast(y_true, tf.float32)
    #y_pred= tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, tf.float32)
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N+K.epsilon()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P+K.epsilon()
 
#接着在模型的compile中设置metrics
#如下例子，我用的是RNN做分类



def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value=tf.zeros(para["val_seed"],name=None)
    update_op=tf.zeros(para["val_seed"],name=None)
    for i in range(len(target)):
        value1, update_op1 = tf.contrib.metrics.streaming_auc(y_pred[:,i], y_true[:,i])
        value=tf.add(value, value1, name=None)
        update_op=tf.add(update_op, update_op1, name=None)
    value=tf.divide(value, len(target), name=None)
    update_op=tf.divide(update_op, len(target), name=None)
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
    
    
    
def densenet121_model(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.5, dropout_rate=para["dropout"], weight_decay=para["weight_decay"], num_classes=None):
    eps = 1.1e-5
    compression = 1.0 - reduction
    global concat_axis
    if K.image_dim_ordering() == 'tf':
        concat_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
        concat_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='data')
    nb_filter = 64
    nb_layers = [6,12,24,16] # For DenseNet-121
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)
    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('sigmoid', name='prob')(x_fc)
    model = Model(img_input, x_fc, name='densenet')
    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'densenet121_weights_th.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'densenet121_weights_tf.h5'
    model.load_weights(weights_path, by_name=True)
##################
#new fine tuning layer
    x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    #x_newfc = Activation('softmax', name='prob')(x_newfc)
    x_newfc = Activation('sigmoid', name='prob')(x_newfc)
    model = Model(img_input, x_newfc)
    # Learning rate is changed to 0.001
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    adam=Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model
def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x
def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)
    return x
def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    eps = 1.1e-5
    concat_feat = x
    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))
        if grow_nb_filters:
            nb_filter += growth_rate
    return concat_feat, nb_filter
if __name__ == '__main__':
    img_rows, img_cols = img_para['dim1'], img_para['dim2']
    channel = 3
    #num_classes = 10 
    num_classes = len(target)
    batch_size = 16 
    nb_epoch = 10
    model = densenet121_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)
    
    



for JJ in disease_list:
    
    bestesttt = np.load(filedir + filename['best'] % JJ)['MM'] +1
    loadingmodel = filedir + filename['model'] % (JJ,bestesttt)
    loadingweight = filedir + filename['weight'] % (JJ,bestesttt)
    
    from keras.models import load_model
    from custom_layers.scale_layer import Scale
    model = load_model(loadingmodel, custom_objects={'Scale': Scale, 'auc_roc': auc_roc, 'auc': auc})
    model.load_weights(loadingweight)
    
    ##predict all test data
    ### data preprocessing ###
    TestList = pd.read_csv(path["test"])
    TestList.set_index('Image Index', inplace=True)
    TestIdList = np.array(TestList.index)
    Id=TestIdList
    ##### predicting ###600800
    # TrainList = pd.read_csv("train.csv")
    # TrainList.set_index('Image Index', inplace=True)
    # TrainList = TrainList['Labels'][TrainList['Labels'].isnull()] #  unlabel data
    # TrainIdList = np.array(TrainList.index)
    # Id=TrainIdList
    #####
    print('Start predicting')    
    Prediction = []
    if img_para['dim1']==100:
        X = np.empty((300,400,3))  
        StartTime = time()
        for i,ID in enumerate(Id):
            img=Image.open(path["img"]+ID)
            region = (100,200,900,800) 
            #Trimming
            cropImg = img.crop(region)
            x = np.array(cropImg)/255
            if x.size != 480000:#x.size != 1048576:
                x = x[:,:,0]
            #x=x.reshape(self.dim, self.dim, 1)
            x=x.reshape(600,800)
            x=skimage.measure.block_reduce(x, (6,6), np.max)
            x=x.reshape(1,100,134,1)
            X=np.repeat(x, 3, axis=3)
            Prediction.append(
                model.predict(
                        X,
                        batch_size = 1,
                        verbose = 0
                        )
                )
            print('predicting:',i,' time elapsed:','{0:.2f}sec'.format(time()-StartTime)) if i % 1000 == 0 else None
    elif img_para['dim1']==300:
        X = np.empty((300,400,3))  
        StartTime = time()
        for i,ID in enumerate(Id):
            img=Image.open(path["img"]+ID)
            region = (100,200,900,800) 
            #Trimming
            cropImg = img.crop(region)
            x = np.array(cropImg)/255
            if x.size != 480000:#x.size != 1048576:
                x = x[:,:,0]
            #x=x.reshape(self.dim, self.dim, 1)
            x=x.reshape(600,800)
            x=skimage.measure.block_reduce(x, (2,2), np.max)
            x=x.reshape(1,300,400,1)
            X=np.repeat(x, 3, axis=3)
            Prediction.append(
                model.predict(
                        X,
                        batch_size = 1,
                        verbose = 0
                        )
                )
            print('predicting:',i,' time elapsed:','{0:.2f}sec'.format(time()-StartTime)) if i % 1000 == 0 else None
    elif img_para['dim1']==600:
        X = np.empty((600,800,3))  
        StartTime = time()
        for i,ID in enumerate(Id):
            img=Image.open(path["img"]+ID)
            region = (100,200,900,800) 
            #Trimming
            cropImg = img.crop(region)
            x = np.array(cropImg)/255
            if x.size != 480000:#x.size != 1048576:
                x = x[:,:,0]
            #x=x.reshape(self.dim, self.dim, 1)
            x=x.reshape(1,600,800,1)
            X=np.repeat(x, 3, axis=3)     
            Prediction.append(
                model.predict(
                        X,
                        batch_size = 1,
                        verbose = 0
                        )
                )
            print('predicting:',i,' time elapsed:','{0:.2f}sec'.format(time()-StartTime)) if i % 1000 == 0 else None
    else: 
        X = np.empty((1024,1024,3))  
        StartTime = time()
        for i,ID in enumerate(Id):
            img=Image.open(path["img"]+ID)
            x = np.array(img)/255
            if x.size != 1024*1024:#x.size != 1048576:
                x = x[:,:,0]
            #x=x.reshape(self.dim, self.dim, 1)
            x=x.reshape(1,1024,1024,1)
            X=np.repeat(x, 3, axis=3)     
            Prediction.append(
                model.predict(
                        X,
                        batch_size = 1,
                        verbose = 0
                        )
                )
            print('predicting:',i,' time elapsed:','{0:.2f}sec'.format(time()-StartTime)) if i % 1000 == 0 else None
    Label = {'category':len(target)}
    Prediction = np.hstack(
            [TestIdList.reshape(-1,1),
             np.array(Prediction).reshape(-1,Label['category'])]       
            )
    ### saving results ###
    print('Saving')
    Sample = pd.read_csv(path["sample"])
    Predict = pd.DataFrame(Prediction)
    Predict.columns = np.hstack((Sample.columns[0],Sample.columns[1:][disease[JJ]]))
    Predict.to_csv(filename['out']%JJ,index=None)  
    
    y_pred = Predict
    try:
        y_predict
    except:
        y_predict = pd.Series(y_pred['Id'])
    y_predict = pd.concat([y_predict,y_pred[JJ]],axis=1)

# =============================================================================
# y_pred = pd.read_csv(filedir+filename['csv']%(JJ,bestesttt,bestesttt))
# try:
#     y_predict
# except:
#     y_predict = pd.Series(y_pred['Id'])
# y_predict = pd.concat([y_predict,y_pred[JJ]],axis=1)
# =============================================================================
    
y_predict.to_csv('../../Submission.csv',index=None)
