import logging
import sys
import numpy as np
from keras.models import Model
from keras.layers import *
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.applications.mobilenet import MobileNet
from keras.utils import plot_model
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
import tensorflow as tf
from capsulelayers import *

from keras.layers.advanced_activations import PReLU
from utils import get_initial_weights
from keras.layers.recurrent import *
from keras.layers.wrappers import *
import loupe_keras as lpk


sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class SSR_net:
    def __init__(self, image_size,stage_num,lambda_local,lambda_d):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)


        self.stage_num = stage_num
        self.lambda_local = lambda_local
        self.lambda_d = lambda_d

    def __call__(self):
        logging.debug("Creating model...")


        inputs = Input(shape=self._input_shape)

        #-------------------------------------------------------------------------------------------------------------------------
        x = Conv2D(32,(3,3))(inputs)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer1 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3))(x_layer1)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer2 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3))(x_layer2)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer3 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3))(x_layer3)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        #-------------------------------------------------------------------------------------------------------------------------
        s = Conv2D(16,(3,3))(inputs)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer1 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3))(s_layer1)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer2 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3))(s_layer2)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer3 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3))(s_layer3)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        

        #-------------------------------------------------------------------------------------------------------------------------
        # Classifier block
        s_layer4 = Conv2D(10,(1,1),activation='relu')(s)
        s_layer4 = Flatten()(s_layer4)
        s_layer4_mix = Dropout(0.2)(s_layer4)
        s_layer4_mix = Dense(units=self.stage_num[0], activation="relu")(s_layer4_mix)
        
        x_layer4 = Conv2D(10,(1,1),activation='relu')(x)
        x_layer4 = Flatten()(x_layer4)
        x_layer4_mix = Dropout(0.2)(x_layer4)
        x_layer4_mix = Dense(units=self.stage_num[0], activation="relu")(x_layer4_mix)
        
        feat_a_s1_pre = Multiply()([s_layer4,x_layer4])
        delta_s1 = Dense(1,activation='tanh',name='delta_s1')(feat_a_s1_pre)
        
        feat_a_s1 = Multiply()([s_layer4_mix,x_layer4_mix])
        feat_a_s1 = Dense(2*self.stage_num[0],activation='relu')(feat_a_s1)
        pred_a_s1 = Dense(units=self.stage_num[0], activation="relu",name='pred_age_stage1')(feat_a_s1)
        #feat_local_s1 = Lambda(lambda x: x/10)(feat_a_s1)
        #feat_a_s1_local = Dropout(0.2)(pred_a_s1)
        local_s1 = Dense(units=self.stage_num[0], activation='tanh', name='local_delta_stage1')(feat_a_s1)
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer2 = Conv2D(10,(1,1),activation='relu')(s_layer2)
        s_layer2 = MaxPooling2D(4,4)(s_layer2)
        s_layer2 = Flatten()(s_layer2)
        s_layer2_mix = Dropout(0.2)(s_layer2)
        s_layer2_mix = Dense(self.stage_num[1],activation='relu')(s_layer2_mix)
        
        x_layer2 = Conv2D(10,(1,1),activation='relu')(x_layer2)
        x_layer2 = AveragePooling2D(4,4)(x_layer2)
        x_layer2 = Flatten()(x_layer2)
        x_layer2_mix = Dropout(0.2)(x_layer2)
        x_layer2_mix = Dense(self.stage_num[1],activation='relu')(x_layer2_mix)
        
        feat_a_s2_pre = Multiply()([s_layer2,x_layer2])
        delta_s2 = Dense(1,activation='tanh',name='delta_s2')(feat_a_s2_pre)
        
        feat_a_s2 = Multiply()([s_layer2_mix,x_layer2_mix])
        feat_a_s2 = Dense(2*self.stage_num[1],activation='relu')(feat_a_s2)
        pred_a_s2 = Dense(units=self.stage_num[1], activation="relu",name='pred_age_stage2')(feat_a_s2)
        #feat_local_s2 = Lambda(lambda x: x/10)(feat_a_s2)
        #feat_a_s2_local = Dropout(0.2)(pred_a_s2)
        local_s2 = Dense(units=self.stage_num[1], activation='tanh', name='local_delta_stage2')(feat_a_s2)
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer1 = Conv2D(10,(1,1),activation='relu')(s_layer1)
        s_layer1 = MaxPooling2D(8,8)(s_layer1)
        s_layer1 = Flatten()(s_layer1)
        s_layer1_mix = Dropout(0.2)(s_layer1)
        s_layer1_mix = Dense(self.stage_num[2],activation='relu')(s_layer1_mix)
        
        x_layer1 = Conv2D(10,(1,1),activation='relu')(x_layer1)
        x_layer1 = AveragePooling2D(8,8)(x_layer1)
        x_layer1 = Flatten()(x_layer1)
        x_layer1_mix = Dropout(0.2)(x_layer1)
        x_layer1_mix = Dense(self.stage_num[2],activation='relu')(x_layer1_mix)

        feat_a_s3_pre = Multiply()([s_layer1,x_layer1])
        delta_s3 = Dense(1,activation='tanh',name='delta_s3')(feat_a_s3_pre)
        
        feat_a_s3 = Multiply()([s_layer1_mix,x_layer1_mix])
        feat_a_s3 = Dense(2*self.stage_num[2],activation='relu')(feat_a_s3)
        pred_a_s3 = Dense(units=self.stage_num[2], activation="relu",name='pred_age_stage3')(feat_a_s3)
        #feat_local_s3 = Lambda(lambda x: x/10)(feat_a_s3)
        #feat_a_s3_local = Dropout(0.2)(pred_a_s3)
        local_s3 = Dense(units=self.stage_num[2], activation='tanh', name='local_delta_stage3')(feat_a_s3)
        #-------------------------------------------------------------------------------------------------------------------------
        
        def merge_age(x,s1,s2,s3,lambda_local,lambda_d):
            a = x[0][:,0]*0
            b = x[0][:,0]*0
            c = x[0][:,0]*0
            A = s1*s2*s3
            V = 101

            for i in range(0,s1):
                a = a+(i+lambda_local*x[6][:,i])*x[0][:,i]
            a = K.expand_dims(a,-1)
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j+lambda_local*x[7][:,j])*x[1][:,j]
            b = K.expand_dims(b,-1)
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k+lambda_local*x[8][:,k])*x[2][:,k]
            c = K.expand_dims(c,-1)
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))


            age = (a+b+c)*V
            return age
        
        pred_a = Lambda(merge_age,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_local':self.lambda_local,'lambda_d':self.lambda_d},name='pred_a')([pred_a_s1,pred_a_s2,pred_a_s3,delta_s1,delta_s2,delta_s3, local_s1, local_s2, local_s3])

        model = Model(inputs=inputs, outputs=pred_a)

        return model
class SSR_net_general:
    def __init__(self, image_size,stage_num,lambda_local,lambda_d):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)


        self.stage_num = stage_num
        self.lambda_local = lambda_local
        self.lambda_d = lambda_d


    def __call__(self):
        logging.debug("Creating model...")


        inputs = Input(shape=self._input_shape)

        #-------------------------------------------------------------------------------------------------------------------------
        x = Conv2D(32,(3,3))(inputs)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer1 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3))(x_layer1)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer2 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3))(x_layer2)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer3 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3))(x_layer3)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        #-------------------------------------------------------------------------------------------------------------------------
        s = Conv2D(16,(3,3))(inputs)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer1 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3))(s_layer1)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer2 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3))(s_layer2)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer3 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3))(s_layer3)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        

        #-------------------------------------------------------------------------------------------------------------------------
        # Classifier block
        s_layer4 = Conv2D(10,(1,1),activation='relu')(s)
        s_layer4 = Flatten()(s_layer4)
        s_layer4_mix = Dropout(0.2)(s_layer4)
        s_layer4_mix = Dense(units=self.stage_num[0], activation="relu")(s_layer4_mix)
        
        x_layer4 = Conv2D(10,(1,1),activation='relu')(x)
        x_layer4 = Flatten()(x_layer4)
        x_layer4_mix = Dropout(0.2)(x_layer4)
        x_layer4_mix = Dense(units=self.stage_num[0], activation="relu")(x_layer4_mix)
        
        feat_s1_pre = Multiply()([s_layer4,x_layer4])
        delta_s1 = Dense(1,activation='tanh',name='delta_s1')(feat_s1_pre)
        
        feat_s1 = Multiply()([s_layer4_mix,x_layer4_mix])
        feat_s1 = Dense(2*self.stage_num[0],activation='relu')(feat_s1)
        pred_s1 = Dense(units=self.stage_num[0], activation="relu",name='pred_stage1')(feat_s1)
        local_s1 = Dense(units=self.stage_num[0], activation='tanh', name='local_delta_stage1')(feat_s1)
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer2 = Conv2D(10,(1,1),activation='relu')(s_layer2)
        s_layer2 = MaxPooling2D(4,4)(s_layer2)
        s_layer2 = Flatten()(s_layer2)
        s_layer2_mix = Dropout(0.2)(s_layer2)
        s_layer2_mix = Dense(self.stage_num[1],activation='relu')(s_layer2_mix)
        
        x_layer2 = Conv2D(10,(1,1),activation='relu')(x_layer2)
        x_layer2 = AveragePooling2D(4,4)(x_layer2)
        x_layer2 = Flatten()(x_layer2)
        x_layer2_mix = Dropout(0.2)(x_layer2)
        x_layer2_mix = Dense(self.stage_num[1],activation='relu')(x_layer2_mix)
        
        feat_s2_pre = Multiply()([s_layer2,x_layer2])
        delta_s2 = Dense(1,activation='tanh',name='delta_s2')(feat_s2_pre)
        
        feat_s2 = Multiply()([s_layer2_mix,x_layer2_mix])
        feat_s2 = Dense(2*self.stage_num[1],activation='relu')(feat_s2)
        pred_s2 = Dense(units=self.stage_num[1], activation="relu",name='pred_stage2')(feat_s2)
        local_s2 = Dense(units=self.stage_num[1], activation='tanh', name='local_delta_stage2')(feat_s2)
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer1 = Conv2D(10,(1,1),activation='relu')(s_layer1)
        s_layer1 = MaxPooling2D(8,8)(s_layer1)
        s_layer1 = Flatten()(s_layer1)
        s_layer1_mix = Dropout(0.2)(s_layer1)
        s_layer1_mix = Dense(self.stage_num[2],activation='relu')(s_layer1_mix)
        
        x_layer1 = Conv2D(10,(1,1),activation='relu')(x_layer1)
        x_layer1 = AveragePooling2D(8,8)(x_layer1)
        x_layer1 = Flatten()(x_layer1)
        x_layer1_mix = Dropout(0.2)(x_layer1)
        x_layer1_mix = Dense(self.stage_num[2],activation='relu')(x_layer1_mix)

        feat_s3_pre = Multiply()([s_layer1,x_layer1])
        delta_s3 = Dense(1,activation='tanh',name='delta_s3')(feat_s3_pre)
        
        feat_s3 = Multiply()([s_layer1_mix,x_layer1_mix])
        feat_s3 = Dense(2*self.stage_num[2],activation='relu')(feat_s3)
        pred_s3 = Dense(units=self.stage_num[2], activation="relu",name='pred_stage3')(feat_s3)
        local_s3 = Dense(units=self.stage_num[2], activation='tanh', name='local_delta_stage3')(feat_s3)
        #-------------------------------------------------------------------------------------------------------------------------
        
        def SSR_module(x,s1,s2,s3,lambda_local,lambda_d):
            a = x[0][:,0]*0
            b = x[0][:,0]*0
            c = x[0][:,0]*0
            V = 1

            for i in range(0,s1):
                a = a+(i+lambda_local*x[6][:,i])*x[0][:,i]
            a = K.expand_dims(a,-1)
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j+lambda_local*x[7][:,j])*x[1][:,j]
            b = K.expand_dims(b,-1)
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k+lambda_local*x[8][:,k])*x[2][:,k]
            c = K.expand_dims(c,-1)
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))


            out = (a+b+c)*V
            return out
        
        pred = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_local':self.lambda_local,'lambda_d':self.lambda_d},name='pred')([pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3, local_s1, local_s2, local_s3])

        model = Model(inputs=inputs, outputs=pred)

        return model
class SSR_net_MT:
    def __init__(self, image_size,num_classes,stage_num,lambda_d):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        #-------------------------------------------------------------------------------------------------------------------------
        x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x_layer1 = AveragePooling2D((2,2))(x)
        x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(32,(3,3),padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x_layer2 = AveragePooling2D((2,2))(x)
        x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(64,(3,3),padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x_layer3 = AveragePooling2D((2,2))(x)
        x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128,(3,3),padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x_layer4 = Activation('relu')(x)
        #-------------------------------------------------------------------------------------------------------------------------
        s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
        s = BatchNormalization(axis=-1)(s)
        s = Activation('tanh')(s)
        s_layer1 = MaxPooling2D((2,2))(s)
        s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
        s = BatchNormalization(axis=-1)(s)
        s = Activation('tanh')(s)
        s = SeparableConv2D(32,(3,3),padding='same')(s)
        s = BatchNormalization(axis=-1)(s)
        s = Activation('tanh')(s)
        s_layer2 = MaxPooling2D((2,2))(s)
        s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
        s = BatchNormalization(axis=-1)(s)
        s = Activation('tanh')(s)
        s = SeparableConv2D(64,(3,3),padding='same')(s)
        s = BatchNormalization(axis=-1)(s)
        s = Activation('tanh')(s)
        s_layer3 = MaxPooling2D((2,2))(s)
        s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
        s = BatchNormalization(axis=-1)(s)
        s = Activation('tanh')(s)
        s = SeparableConv2D(128,(3,3),padding='same')(s)
        s = BatchNormalization(axis=-1)(s)
        s_layer4 = Activation('tanh')(s)


        #-------------------------------------------------------------------------------------------------------------------------
        # Classifier block
        s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
        s_layer4 = MaxPooling2D((2,2))(s_layer4)

        x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
        x_layer4 = AveragePooling2D((2,2))(x_layer4)

        feat_s1_pre = Multiply()([s_layer4,x_layer4])
        feat_s1_pre = Flatten()(feat_s1_pre)
        feat_delta_s1 = Dense(2*self.num_classes,activation='tanh')(feat_s1_pre)
        delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

        feat_local_s1 = Dense(2*self.num_classes,activation='tanh')(feat_s1_pre)
        local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

        feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(feat_s1_pre) 
        pred_a_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
        s_layer3 = MaxPooling2D((2,2))(s_layer3)

        x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
        x_layer3 = AveragePooling2D((2,2))(x_layer3)

        feat_s2_pre = Multiply()([s_layer3,x_layer3])
        feat_s2_pre  = Flatten()(feat_s2_pre)
        feat_delta_s2 = Dense(2*self.num_classes,activation='tanh')(feat_s2_pre)
        delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

        feat_local_s2 = Dense(2*self.num_classes,activation='tanh')(feat_s2_pre)
        local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

        feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(feat_s2_pre) 
        pred_a_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
        s_layer2 = MaxPooling2D((2,2))(s_layer2)

        x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
        x_layer2 = AveragePooling2D((2,2))(x_layer2)

        feat_s3_pre = Multiply()([s_layer2,x_layer2])
        feat_s3_pre  = Flatten()(feat_s3_pre)
        feat_delta_s3 = Dense(2*self.num_classes,activation='tanh')(feat_s3_pre)
        delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

        feat_local_s3 = Dense(2*self.num_classes,activation='tanh')(feat_s3_pre)
        local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

        feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(feat_s3_pre) 
        pred_a_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            # a = K.expand_dims(a,-1)
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            # b = K.expand_dims(b,-1)
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            # c = K.expand_dims(c,-1)
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')([pred_a_s1,pred_a_s2,pred_a_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3])

        model = Model(inputs=img_inputs, outputs=pred_pose)

        return model
class SSR_net_ori_MT:
    def __init__(self, image_size,num_classes,stage_num,lambda_d):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        #-------------------------------------------------------------------------------------------------------------------------
        x = Conv2D(32,(3,3),padding='same')(img_inputs)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer1 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3),padding='same')(x_layer1)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer2 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3),padding='same')(x_layer2)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x = Activation('relu')(x)
        x_layer3 = AveragePooling2D(2,2)(x)
        x = Conv2D(32,(3,3),padding='same')(x_layer3)
        x = BatchNormalization(axis=self._channel_axis)(x)
        x_layer4 = Activation('relu')(x)
        #-------------------------------------------------------------------------------------------------------------------------
        s = Conv2D(16,(3,3),padding='same')(img_inputs)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer1 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3),padding='same')(s_layer1)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer2 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3),padding='same')(s_layer2)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s = Activation('tanh')(s)
        s_layer3 = MaxPooling2D(2,2)(s)
        s = Conv2D(16,(3,3),padding='same')(s_layer3)
        s = BatchNormalization(axis=self._channel_axis)(s)
        s_layer4 = Activation('tanh')(s)
        
        #-------------------------------------------------------------------------------------------------------------------------
        # Classifier block
        s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
        s_layer4 = MaxPooling2D((2,2))(s_layer4)

        x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
        x_layer4 = AveragePooling2D((2,2))(x_layer4)

        feat_s1_pre = Multiply()([s_layer4,x_layer4])
        feat_s1_pre = Flatten()(feat_s1_pre)
        feat_delta_s1 = Dense(2*self.num_classes,activation='tanh')(feat_s1_pre)
        delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

        feat_local_s1 = Dense(2*self.num_classes,activation='tanh')(feat_s1_pre)
        local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

        feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(feat_s1_pre) 
        pred_a_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
        s_layer3 = MaxPooling2D((2,2))(s_layer3)

        x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
        x_layer3 = AveragePooling2D((2,2))(x_layer3)

        feat_s2_pre = Multiply()([s_layer3,x_layer3])
        feat_s2_pre  = Flatten()(feat_s2_pre)
        feat_delta_s2 = Dense(2*self.num_classes,activation='tanh')(feat_s2_pre)
        delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

        feat_local_s2 = Dense(2*self.num_classes,activation='tanh')(feat_s2_pre)
        local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

        feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(feat_s2_pre) 
        pred_a_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
        s_layer2 = MaxPooling2D((2,2))(s_layer2)

        x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
        x_layer2 = AveragePooling2D((2,2))(x_layer2)

        feat_s3_pre = Multiply()([s_layer2,x_layer2])
        feat_s3_pre  = Flatten()(feat_s3_pre)
        feat_delta_s3 = Dense(2*self.num_classes,activation='tanh')(feat_s3_pre)
        delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

        feat_local_s3 = Dense(2*self.num_classes,activation='tanh')(feat_s3_pre)
        local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

        feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(feat_s3_pre) 
        pred_a_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            # a = K.expand_dims(a,-1)
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            # b = K.expand_dims(b,-1)
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            # c = K.expand_dims(c,-1)
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')([pred_a_s1,pred_a_s2,pred_a_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3])

        model = Model(inputs=img_inputs, outputs=pred_pose)

        return model



class FSA_net_Capsule:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

        
    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        def ssr_feat_S_model_build(num_primcaps, m_dim):
            input_preS = Input((8,8,64))

            feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
            feat_preS = Reshape((-1,))(feat_preS)
            SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
            SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
            
            ssr_feat_S_model = Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')
            return ssr_feat_S_model

        ssr_feat_S_model = ssr_feat_S_model_build(self.num_primcaps,self.m_dim)  
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build(num_primcaps, m_dim):
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            SR_matrix_s1,feat_s1_preS = ssr_feat_S_model(input_s1_preS)
            SR_matrix_s2,feat_s2_preS = ssr_feat_S_model(input_s2_preS)
            SR_matrix_s3,feat_s3_preS = ssr_feat_S_model(input_s3_preS)
            
            feat_pre_concat = Concatenate()([feat_s1_preS,feat_s2_preS,feat_s3_preS])
            SL_matrix = Dense(int(num_primcaps/3)*m_dim,activation='sigmoid')(feat_pre_concat)
            SL_matrix = Reshape((int(num_primcaps/3),m_dim))(SL_matrix)
            
            S_matrix_s1 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s1')([SL_matrix,SR_matrix_s1])
            S_matrix_s2 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s2')([SL_matrix,SR_matrix_s2])
            S_matrix_s3 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s3')([SL_matrix,SR_matrix_s3])

            # Very important!!! Without this training won't converge.
            # norm_S = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix)
            norm_S_s1 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s1)
            norm_S_s2 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s2)
            norm_S_s3 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s3)

            feat_s1_pre = Reshape((-1,64))(input_s1_preS)
            feat_s2_pre = Reshape((-1,64))(input_s2_preS)
            feat_s3_pre = Reshape((-1,64))(input_s3_preS)
            feat_pre_concat = Concatenate(axis=1)([feat_s1_pre, feat_s2_pre, feat_s3_pre])
            
            # Warining: don't use keras's 'K.dot'. It is very weird when high dimension is used.
            # https://github.com/keras-team/keras/issues/9779
            # Make sure 'tf.matmul' is used
            # primcaps = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix,feat_pre_concat, norm_S])
            primcaps_s1 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s1,feat_pre_concat, norm_S_s1])
            primcaps_s2 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s2,feat_pre_concat, norm_S_s2])
            primcaps_s3 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s3,feat_pre_concat, norm_S_s3])
            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build(self.num_primcaps,self.m_dim)        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Cap_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)


            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(capsule)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(capsule)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(capsule)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Cap_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Cap_model')            
            return ssr_Cap_model

        ssr_Cap_model = ssr_Cap_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Lambda(lambda x: x[:,0:4])(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Lambda(lambda x: x[:,4:8])(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Lambda(lambda x: x[:,8:16])(input_s1_pre)
            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(feat_pred_s1) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Lambda(lambda x: x[:,0:4])(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Lambda(lambda x: x[:,4:8])(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Lambda(lambda x: x[:,8:16])(input_s2_pre)
            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(feat_pred_s2) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Lambda(lambda x: x[:,0:4])(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Lambda(lambda x: x[:,4:8])(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Lambda(lambda x: x[:,8:16])(input_s3_pre)
            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(feat_pred_s3) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
        
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        #ssr_F_model = ssr_F_model_build(21*64,'ssr_F_model')
        # ssr_F_Cap_model = ssr_F_model_build(int(self.dim_capsule/3)*self.num_capsule,'ssr_F_model_Cap')
        ssr_F_Cap_model = ssr_F_model_build(self.F_shape,'ssr_F_Cap_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Cap_list = ssr_Cap_model(ssr_primcaps)
        ssr_F_Cap_list = ssr_F_Cap_model(ssr_Cap_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_Cap_list)
        

        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model
class FSA_net_Var_Capsule:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

        
    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        def ssr_feat_S_model_build(num_primcaps, m_dim):
            input_preS = Input((8,8,64))

            #feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
            def var(x):
                mean, var = tf.nn.moments(x,axes=-1)
                return var
            feat_preS = Lambda(var)(input_preS)
            feat_preS = Reshape((-1,))(feat_preS)
            SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
            SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
            
            ssr_feat_S_model = Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')
            return ssr_feat_S_model

        ssr_feat_S_model = ssr_feat_S_model_build(self.num_primcaps,self.m_dim)  
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build(num_primcaps, m_dim):
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            SR_matrix_s1,feat_s1_preS = ssr_feat_S_model(input_s1_preS)
            SR_matrix_s2,feat_s2_preS = ssr_feat_S_model(input_s2_preS)
            SR_matrix_s3,feat_s3_preS = ssr_feat_S_model(input_s3_preS)
            
            feat_pre_concat = Concatenate()([feat_s1_preS,feat_s2_preS,feat_s3_preS])
            SL_matrix = Dense(int(num_primcaps/3)*m_dim,activation='sigmoid')(feat_pre_concat)
            SL_matrix = Reshape((int(num_primcaps/3),m_dim))(SL_matrix)
            
            S_matrix_s1 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s1')([SL_matrix,SR_matrix_s1])
            S_matrix_s2 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s2')([SL_matrix,SR_matrix_s2])
            S_matrix_s3 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s3')([SL_matrix,SR_matrix_s3])

            # Very important!!! Without this training won't converge.
            # norm_S = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix)
            norm_S_s1 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s1)
            norm_S_s2 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s2)
            norm_S_s3 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s3)

            feat_s1_pre = Reshape((-1,64))(input_s1_preS)
            feat_s2_pre = Reshape((-1,64))(input_s2_preS)
            feat_s3_pre = Reshape((-1,64))(input_s3_preS)
            feat_pre_concat = Concatenate(axis=1)([feat_s1_pre, feat_s2_pre, feat_s3_pre])
            
            # Warining: don't use keras's 'K.dot'. It is very weird when high dimension is used.
            # https://github.com/keras-team/keras/issues/9779
            # Make sure 'tf.matmul' is used
            # primcaps = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix,feat_pre_concat, norm_S])
            primcaps_s1 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s1,feat_pre_concat, norm_S_s1])
            primcaps_s2 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s2,feat_pre_concat, norm_S_s2])
            primcaps_s3 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s3,feat_pre_concat, norm_S_s3])
            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build(self.num_primcaps,self.m_dim)        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Cap_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)


            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(capsule)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(capsule)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(capsule)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Cap_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Cap_model')            
            return ssr_Cap_model

        ssr_Cap_model = ssr_Cap_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Lambda(lambda x: x[:,0:4])(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Lambda(lambda x: x[:,4:8])(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Lambda(lambda x: x[:,8:16])(input_s1_pre)
            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(feat_pred_s1) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Lambda(lambda x: x[:,0:4])(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Lambda(lambda x: x[:,4:8])(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Lambda(lambda x: x[:,8:16])(input_s2_pre)
            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(feat_pred_s2) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Lambda(lambda x: x[:,0:4])(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Lambda(lambda x: x[:,4:8])(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Lambda(lambda x: x[:,8:16])(input_s3_pre)
            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(feat_pred_s3) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
        
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        #ssr_F_model = ssr_F_model_build(21*64,'ssr_F_model')
        # ssr_F_Cap_model = ssr_F_model_build(int(self.dim_capsule/3)*self.num_capsule,'ssr_F_model_Cap')
        ssr_F_Cap_model = ssr_F_model_build(self.F_shape,'ssr_F_Cap_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Cap_list = ssr_Cap_model(ssr_primcaps)
        ssr_F_Cap_list = ssr_F_Cap_model(ssr_Cap_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_Cap_list)
        

        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model
class FSA_net_noS_Capsule:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

        
    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build():
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            primcaps_s1 = Reshape((8*8,64))(input_s1_preS)
            primcaps_s2 = Reshape((8*8,64))(input_s2_preS)
            primcaps_s3 = Reshape((8*8,64))(input_s3_preS)

            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build()        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Cap_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)


            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(capsule)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(capsule)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(capsule)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Cap_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Cap_model')            
            return ssr_Cap_model

        ssr_Cap_model = ssr_Cap_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Lambda(lambda x: x[:,0:4])(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Lambda(lambda x: x[:,4:8])(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Lambda(lambda x: x[:,8:16])(input_s1_pre)
            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(feat_pred_s1) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Lambda(lambda x: x[:,0:4])(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Lambda(lambda x: x[:,4:8])(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Lambda(lambda x: x[:,8:16])(input_s2_pre)
            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(feat_pred_s2) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Lambda(lambda x: x[:,0:4])(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Lambda(lambda x: x[:,4:8])(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Lambda(lambda x: x[:,8:16])(input_s3_pre)
            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(feat_pred_s3) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
        
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        ssr_F_Cap_model = ssr_F_model_build(self.F_shape,'ssr_F_Cap_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Cap_list = ssr_Cap_model(ssr_primcaps)
        ssr_F_Cap_list = ssr_F_Cap_model(ssr_Cap_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_Cap_list)
        

        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model



class FSA_net_Capsule_FC:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

        
    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        def ssr_feat_S_model_build(num_primcaps, m_dim):
            input_preS = Input((8,8,64))

            feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
            feat_preS = Reshape((-1,))(feat_preS)
            SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
            SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
            
            ssr_feat_S_model = Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')
            return ssr_feat_S_model

        ssr_feat_S_model = ssr_feat_S_model_build(self.num_primcaps,self.m_dim)  
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build(num_primcaps, m_dim):
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            SR_matrix_s1,feat_s1_preS = ssr_feat_S_model(input_s1_preS)
            SR_matrix_s2,feat_s2_preS = ssr_feat_S_model(input_s2_preS)
            SR_matrix_s3,feat_s3_preS = ssr_feat_S_model(input_s3_preS)
            
            feat_pre_concat = Concatenate()([feat_s1_preS,feat_s2_preS,feat_s3_preS])
            SL_matrix = Dense(int(num_primcaps/3)*m_dim,activation='sigmoid')(feat_pre_concat)
            SL_matrix = Reshape((int(num_primcaps/3),m_dim))(SL_matrix)
            
            S_matrix_s1 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s1')([SL_matrix,SR_matrix_s1])
            S_matrix_s2 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s2')([SL_matrix,SR_matrix_s2])
            S_matrix_s3 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s3')([SL_matrix,SR_matrix_s3])

            # Very important!!! Without this training won't converge.
            # norm_S = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix)
            norm_S_s1 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s1)
            norm_S_s2 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s2)
            norm_S_s3 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s3)

            feat_s1_pre = Reshape((-1,64))(input_s1_preS)
            feat_s2_pre = Reshape((-1,64))(input_s2_preS)
            feat_s3_pre = Reshape((-1,64))(input_s3_preS)
            feat_pre_concat = Concatenate(axis=1)([feat_s1_pre, feat_s2_pre, feat_s3_pre])
            
            # Warining: don't use keras's 'K.dot'. It is very weird when high dimension is used.
            # https://github.com/keras-team/keras/issues/9779
            # Make sure 'tf.matmul' is used
            # primcaps = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix,feat_pre_concat, norm_S])
            primcaps_s1 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s1,feat_pre_concat, norm_S_s1])
            primcaps_s2 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s2,feat_pre_concat, norm_S_s2])
            primcaps_s3 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s3,feat_pre_concat, norm_S_s3])
            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build(self.num_primcaps,self.m_dim)        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Cap_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)


            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(capsule)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(capsule)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(capsule)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Cap_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Cap_model')            
            return ssr_Cap_model

        ssr_Cap_model = ssr_Cap_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Dense(2*self.num_classes,activation='tanh')(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Dense(2*self.num_classes,activation='tanh')(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(input_s1_pre) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Dense(2*self.num_classes,activation='tanh')(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Dense(2*self.num_classes,activation='tanh')(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(input_s2_pre) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Dense(2*self.num_classes,activation='tanh')(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Dense(2*self.num_classes,activation='tanh')(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(input_s3_pre) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
        
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        #ssr_F_model = ssr_F_model_build(21*64,'ssr_F_model')
        # ssr_F_Cap_model = ssr_F_model_build(int(self.dim_capsule/3)*self.num_capsule,'ssr_F_model_Cap')
        ssr_F_Cap_model = ssr_F_model_build(self.F_shape,'ssr_F_Cap_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Cap_list = ssr_Cap_model(ssr_primcaps)
        ssr_F_Cap_list = ssr_F_Cap_model(ssr_Cap_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_Cap_list)
        

        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model
class FSA_net_Var_Capsule_FC:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

        
    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        def ssr_feat_S_model_build(num_primcaps, m_dim):
            input_preS = Input((8,8,64))

            #feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
            def var(x):
                mean, var = tf.nn.moments(x,axes=-1)
                return var
            feat_preS = Lambda(var)(input_preS)
            feat_preS = Reshape((-1,))(feat_preS)
            SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
            SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
            
            ssr_feat_S_model = Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')
            return ssr_feat_S_model

        ssr_feat_S_model = ssr_feat_S_model_build(self.num_primcaps,self.m_dim)  
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build(num_primcaps, m_dim):
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            SR_matrix_s1,feat_s1_preS = ssr_feat_S_model(input_s1_preS)
            SR_matrix_s2,feat_s2_preS = ssr_feat_S_model(input_s2_preS)
            SR_matrix_s3,feat_s3_preS = ssr_feat_S_model(input_s3_preS)
            
            feat_pre_concat = Concatenate()([feat_s1_preS,feat_s2_preS,feat_s3_preS])
            SL_matrix = Dense(int(num_primcaps/3)*m_dim,activation='sigmoid')(feat_pre_concat)
            SL_matrix = Reshape((int(num_primcaps/3),m_dim))(SL_matrix)
            
            S_matrix_s1 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s1')([SL_matrix,SR_matrix_s1])
            S_matrix_s2 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s2')([SL_matrix,SR_matrix_s2])
            S_matrix_s3 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s3')([SL_matrix,SR_matrix_s3])

            # Very important!!! Without this training won't converge.
            # norm_S = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix)
            norm_S_s1 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s1)
            norm_S_s2 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s2)
            norm_S_s3 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s3)

            feat_s1_pre = Reshape((-1,64))(input_s1_preS)
            feat_s2_pre = Reshape((-1,64))(input_s2_preS)
            feat_s3_pre = Reshape((-1,64))(input_s3_preS)
            feat_pre_concat = Concatenate(axis=1)([feat_s1_pre, feat_s2_pre, feat_s3_pre])
            
            # Warining: don't use keras's 'K.dot'. It is very weird when high dimension is used.
            # https://github.com/keras-team/keras/issues/9779
            # Make sure 'tf.matmul' is used
            # primcaps = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix,feat_pre_concat, norm_S])
            primcaps_s1 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s1,feat_pre_concat, norm_S_s1])
            primcaps_s2 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s2,feat_pre_concat, norm_S_s2])
            primcaps_s3 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s3,feat_pre_concat, norm_S_s3])
            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build(self.num_primcaps,self.m_dim)        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Cap_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)


            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(capsule)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(capsule)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(capsule)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Cap_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Cap_model')            
            return ssr_Cap_model

        ssr_Cap_model = ssr_Cap_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Dense(2*self.num_classes,activation='tanh')(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Dense(2*self.num_classes,activation='tanh')(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(input_s1_pre) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Dense(2*self.num_classes,activation='tanh')(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Dense(2*self.num_classes,activation='tanh')(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(input_s2_pre) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Dense(2*self.num_classes,activation='tanh')(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Dense(2*self.num_classes,activation='tanh')(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(input_s3_pre) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
       
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        #ssr_F_model = ssr_F_model_build(21*64,'ssr_F_model')
        # ssr_F_Cap_model = ssr_F_model_build(int(self.dim_capsule/3)*self.num_capsule,'ssr_F_model_Cap')
        ssr_F_Cap_model = ssr_F_model_build(self.F_shape,'ssr_F_Cap_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Cap_list = ssr_Cap_model(ssr_primcaps)
        ssr_F_Cap_list = ssr_F_Cap_model(ssr_Cap_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_Cap_list)
        

        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model
class FSA_net_noS_Capsule_FC:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

        
    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build():
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            primcaps_s1 = Reshape((8*8,64))(input_s1_preS)
            primcaps_s2 = Reshape((8*8,64))(input_s2_preS)
            primcaps_s3 = Reshape((8*8,64))(input_s3_preS)

            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build()        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Cap_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)


            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(capsule)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(capsule)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(capsule)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Cap_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Cap_model')            
            return ssr_Cap_model

        ssr_Cap_model = ssr_Cap_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Dense(2*self.num_classes,activation='tanh')(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Dense(2*self.num_classes,activation='tanh')(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(input_s1_pre) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Dense(2*self.num_classes,activation='tanh')(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Dense(2*self.num_classes,activation='tanh')(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(input_s2_pre) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Dense(2*self.num_classes,activation='tanh')(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Dense(2*self.num_classes,activation='tanh')(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(input_s3_pre) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
       
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        ssr_F_Cap_model = ssr_F_model_build(self.F_shape,'ssr_F_Cap_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Cap_list = ssr_Cap_model(ssr_primcaps)
        ssr_F_Cap_list = ssr_F_Cap_model(ssr_Cap_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_Cap_list)
        

        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model


class FSA_net_NetVLAD:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        def ssr_feat_S_model_build(num_primcaps, m_dim):
            input_preS = Input((8,8,64))

            feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
            feat_preS = Reshape((-1,))(feat_preS)
            SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
            SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
            
            ssr_feat_S_model = Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')
            return ssr_feat_S_model

        ssr_feat_S_model = ssr_feat_S_model_build(self.num_primcaps,self.m_dim)  
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build(num_primcaps, m_dim):
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            SR_matrix_s1,feat_s1_preS = ssr_feat_S_model(input_s1_preS)
            SR_matrix_s2,feat_s2_preS = ssr_feat_S_model(input_s2_preS)
            SR_matrix_s3,feat_s3_preS = ssr_feat_S_model(input_s3_preS)
            
            feat_pre_concat = Concatenate()([feat_s1_preS,feat_s2_preS,feat_s3_preS])
            SL_matrix = Dense(int(num_primcaps/3)*m_dim,activation='sigmoid')(feat_pre_concat)
            SL_matrix = Reshape((int(num_primcaps/3),m_dim))(SL_matrix)
            
            S_matrix_s1 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s1')([SL_matrix,SR_matrix_s1])
            S_matrix_s2 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s2')([SL_matrix,SR_matrix_s2])
            S_matrix_s3 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s3')([SL_matrix,SR_matrix_s3])

            # Very important!!! Without this training won't converge.
            # norm_S = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix)
            norm_S_s1 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s1)
            norm_S_s2 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s2)
            norm_S_s3 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s3)

            feat_s1_pre = Reshape((-1,64))(input_s1_preS)
            feat_s2_pre = Reshape((-1,64))(input_s2_preS)
            feat_s3_pre = Reshape((-1,64))(input_s3_preS)
            feat_pre_concat = Concatenate(axis=1)([feat_s1_pre, feat_s2_pre, feat_s3_pre])
            
            # Warining: don't use keras's 'K.dot'. It is very weird when high dimension is used.
            # https://github.com/keras-team/keras/issues/9779
            # Make sure 'tf.matmul' is used
            # primcaps = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix,feat_pre_concat, norm_S])
            primcaps_s1 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s1,feat_pre_concat, norm_S_s1])
            primcaps_s2 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s2,feat_pre_concat, norm_S_s2])
            primcaps_s3 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s3,feat_pre_concat, norm_S_s3])
            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build(self.num_primcaps,self.m_dim)        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Agg_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            #capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)
            agg_feat = lpk.NetVLAD(feature_size=64, max_samples=self.num_primcaps, cluster_size=self.num_capsule, output_dim=self.num_capsule*self.dim_capsule)(input_primcaps)
            agg_feat = Reshape((self.num_capsule,self.dim_capsule))(agg_feat)

            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(agg_feat)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(agg_feat)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(agg_feat)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Agg_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Agg_model')            
            return ssr_Agg_model

        ssr_Agg_model = ssr_Agg_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Lambda(lambda x: x[:,0:4])(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Lambda(lambda x: x[:,4:8])(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Lambda(lambda x: x[:,8:16])(input_s1_pre)
            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(feat_pred_s1) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Lambda(lambda x: x[:,0:4])(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Lambda(lambda x: x[:,4:8])(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Lambda(lambda x: x[:,8:16])(input_s2_pre)
            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(feat_pred_s2) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Lambda(lambda x: x[:,0:4])(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Lambda(lambda x: x[:,4:8])(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Lambda(lambda x: x[:,8:16])(input_s3_pre)
            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(feat_pred_s3) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
        
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        ssr_F_model = ssr_F_model_build(self.F_shape,'ssr_F_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Agg_list = ssr_Agg_model(ssr_primcaps)
        ssr_F_list = ssr_F_model(ssr_Agg_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_list)
        

        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model
class FSA_net_Var_NetVLAD:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

            
    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        def ssr_feat_S_model_build(num_primcaps, m_dim):
            input_preS = Input((8,8,64))

            #feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
            def var(x):
                mean, var = tf.nn.moments(x,axes=-1)
                return var
            feat_preS = Lambda(var)(input_preS)
            feat_preS = Reshape((-1,))(feat_preS)
            SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
            SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
            
            ssr_feat_S_model = Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')
            return ssr_feat_S_model

        ssr_feat_S_model = ssr_feat_S_model_build(self.num_primcaps,self.m_dim)  
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build(num_primcaps, m_dim):
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            SR_matrix_s1,feat_s1_preS = ssr_feat_S_model(input_s1_preS)
            SR_matrix_s2,feat_s2_preS = ssr_feat_S_model(input_s2_preS)
            SR_matrix_s3,feat_s3_preS = ssr_feat_S_model(input_s3_preS)
            
            feat_pre_concat = Concatenate()([feat_s1_preS,feat_s2_preS,feat_s3_preS])
            SL_matrix = Dense(int(num_primcaps/3)*m_dim,activation='sigmoid')(feat_pre_concat)
            SL_matrix = Reshape((int(num_primcaps/3),m_dim))(SL_matrix)
            
            S_matrix_s1 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s1')([SL_matrix,SR_matrix_s1])
            S_matrix_s2 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s2')([SL_matrix,SR_matrix_s2])
            S_matrix_s3 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s3')([SL_matrix,SR_matrix_s3])

            # Very important!!! Without this training won't converge.
            # norm_S = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix)
            norm_S_s1 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s1)
            norm_S_s2 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s2)
            norm_S_s3 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s3)

            feat_s1_pre = Reshape((-1,64))(input_s1_preS)
            feat_s2_pre = Reshape((-1,64))(input_s2_preS)
            feat_s3_pre = Reshape((-1,64))(input_s3_preS)
            feat_pre_concat = Concatenate(axis=1)([feat_s1_pre, feat_s2_pre, feat_s3_pre])
            
            # Warining: don't use keras's 'K.dot'. It is very weird when high dimension is used.
            # https://github.com/keras-team/keras/issues/9779
            # Make sure 'tf.matmul' is used
            # primcaps = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix,feat_pre_concat, norm_S])
            primcaps_s1 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s1,feat_pre_concat, norm_S_s1])
            primcaps_s2 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s2,feat_pre_concat, norm_S_s2])
            primcaps_s3 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s3,feat_pre_concat, norm_S_s3])
            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build(self.num_primcaps,self.m_dim)        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Agg_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            #capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)
            agg_feat = lpk.NetVLAD(feature_size=64, max_samples=self.num_primcaps, cluster_size=self.num_capsule, output_dim=self.num_capsule*self.dim_capsule)(input_primcaps)
            agg_feat = Reshape((self.num_capsule,self.dim_capsule))(agg_feat)

            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(agg_feat)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(agg_feat)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(agg_feat)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Agg_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Agg_model')            
            return ssr_Agg_model

        ssr_Agg_model = ssr_Agg_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Lambda(lambda x: x[:,0:4])(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Lambda(lambda x: x[:,4:8])(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Lambda(lambda x: x[:,8:16])(input_s1_pre)
            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(feat_pred_s1) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Lambda(lambda x: x[:,0:4])(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Lambda(lambda x: x[:,4:8])(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Lambda(lambda x: x[:,8:16])(input_s2_pre)
            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(feat_pred_s2) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Lambda(lambda x: x[:,0:4])(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Lambda(lambda x: x[:,4:8])(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Lambda(lambda x: x[:,8:16])(input_s3_pre)
            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(feat_pred_s3) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
        
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        ssr_F_model = ssr_F_model_build(self.F_shape,'ssr_F_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Agg_list = ssr_Agg_model(ssr_primcaps)
        ssr_F_list = ssr_F_model(ssr_Agg_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_list)
        

        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model
class FSA_net_noS_NetVLAD:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

            
    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build():
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            primcaps_s1 = Reshape((8*8,64))(input_s1_preS)
            primcaps_s2 = Reshape((8*8,64))(input_s2_preS)
            primcaps_s3 = Reshape((8*8,64))(input_s3_preS)
            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build()        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Agg_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            #capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)
            agg_feat = lpk.NetVLAD(feature_size=64, max_samples=self.num_primcaps, cluster_size=self.num_capsule, output_dim=self.num_capsule*self.dim_capsule)(input_primcaps)
            agg_feat = Reshape((self.num_capsule,self.dim_capsule))(agg_feat)

            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(agg_feat)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(agg_feat)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(agg_feat)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Agg_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Agg_model')            
            return ssr_Agg_model

        ssr_Agg_model = ssr_Agg_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Lambda(lambda x: x[:,0:4])(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Lambda(lambda x: x[:,4:8])(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Lambda(lambda x: x[:,8:16])(input_s1_pre)
            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(feat_pred_s1) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Lambda(lambda x: x[:,0:4])(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Lambda(lambda x: x[:,4:8])(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Lambda(lambda x: x[:,8:16])(input_s2_pre)
            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(feat_pred_s2) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Lambda(lambda x: x[:,0:4])(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Lambda(lambda x: x[:,4:8])(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Lambda(lambda x: x[:,8:16])(input_s3_pre)
            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(feat_pred_s3) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
        
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        ssr_F_model = ssr_F_model_build(self.F_shape,'ssr_F_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Agg_list = ssr_Agg_model(ssr_primcaps)
        ssr_F_list = ssr_F_model(ssr_Agg_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_list)
        

        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model


class FSA_net_NetVLAD_FC:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        def ssr_feat_S_model_build(num_primcaps, m_dim):
            input_preS = Input((8,8,64))

            feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
            feat_preS = Reshape((-1,))(feat_preS)
            SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
            SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
            
            ssr_feat_S_model = Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')
            return ssr_feat_S_model

        ssr_feat_S_model = ssr_feat_S_model_build(self.num_primcaps,self.m_dim)  
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build(num_primcaps, m_dim):
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            SR_matrix_s1,feat_s1_preS = ssr_feat_S_model(input_s1_preS)
            SR_matrix_s2,feat_s2_preS = ssr_feat_S_model(input_s2_preS)
            SR_matrix_s3,feat_s3_preS = ssr_feat_S_model(input_s3_preS)
            
            feat_pre_concat = Concatenate()([feat_s1_preS,feat_s2_preS,feat_s3_preS])
            SL_matrix = Dense(int(num_primcaps/3)*m_dim,activation='sigmoid')(feat_pre_concat)
            SL_matrix = Reshape((int(num_primcaps/3),m_dim))(SL_matrix)
            
            S_matrix_s1 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s1')([SL_matrix,SR_matrix_s1])
            S_matrix_s2 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s2')([SL_matrix,SR_matrix_s2])
            S_matrix_s3 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s3')([SL_matrix,SR_matrix_s3])

            # Very important!!! Without this training won't converge.
            # norm_S = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix)
            norm_S_s1 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s1)
            norm_S_s2 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s2)
            norm_S_s3 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s3)

            feat_s1_pre = Reshape((-1,64))(input_s1_preS)
            feat_s2_pre = Reshape((-1,64))(input_s2_preS)
            feat_s3_pre = Reshape((-1,64))(input_s3_preS)
            feat_pre_concat = Concatenate(axis=1)([feat_s1_pre, feat_s2_pre, feat_s3_pre])
            
            # Warining: don't use keras's 'K.dot'. It is very weird when high dimension is used.
            # https://github.com/keras-team/keras/issues/9779
            # Make sure 'tf.matmul' is used
            # primcaps = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix,feat_pre_concat, norm_S])
            primcaps_s1 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s1,feat_pre_concat, norm_S_s1])
            primcaps_s2 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s2,feat_pre_concat, norm_S_s2])
            primcaps_s3 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s3,feat_pre_concat, norm_S_s3])
            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build(self.num_primcaps,self.m_dim)        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Agg_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            #capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)
            agg_feat = lpk.NetVLAD(feature_size=64, max_samples=self.num_primcaps, cluster_size=self.num_capsule, output_dim=self.num_capsule*self.dim_capsule)(input_primcaps)
            agg_feat = Reshape((self.num_capsule,self.dim_capsule))(agg_feat)

            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(agg_feat)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(agg_feat)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(agg_feat)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Agg_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Agg_model')            
            return ssr_Agg_model

        ssr_Agg_model = ssr_Agg_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Dense(2*self.num_classes,activation='tanh')(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Dense(2*self.num_classes,activation='tanh')(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(input_s1_pre) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Dense(2*self.num_classes,activation='tanh')(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Dense(2*self.num_classes,activation='tanh')(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(input_s2_pre) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Dense(2*self.num_classes,activation='tanh')(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Dense(2*self.num_classes,activation='tanh')(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(input_s3_pre) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
       
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        ssr_F_model = ssr_F_model_build(self.F_shape,'ssr_F_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Agg_list = ssr_Agg_model(ssr_primcaps)
        ssr_F_list = ssr_F_model(ssr_Agg_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_list)
        

        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model
class FSA_net_Var_NetVLAD_FC:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

            
    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        def ssr_feat_S_model_build(num_primcaps, m_dim):
            input_preS = Input((8,8,64))

            #feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
            def var(x):
                mean, var = tf.nn.moments(x,axes=-1)
                return var
            feat_preS = Lambda(var)(input_preS)
            feat_preS = Reshape((-1,))(feat_preS)
            SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
            SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
            
            ssr_feat_S_model = Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')
            return ssr_feat_S_model

        ssr_feat_S_model = ssr_feat_S_model_build(self.num_primcaps,self.m_dim)  
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build(num_primcaps, m_dim):
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            SR_matrix_s1,feat_s1_preS = ssr_feat_S_model(input_s1_preS)
            SR_matrix_s2,feat_s2_preS = ssr_feat_S_model(input_s2_preS)
            SR_matrix_s3,feat_s3_preS = ssr_feat_S_model(input_s3_preS)
            
            feat_pre_concat = Concatenate()([feat_s1_preS,feat_s2_preS,feat_s3_preS])
            SL_matrix = Dense(int(num_primcaps/3)*m_dim,activation='sigmoid')(feat_pre_concat)
            SL_matrix = Reshape((int(num_primcaps/3),m_dim))(SL_matrix)
            
            S_matrix_s1 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s1')([SL_matrix,SR_matrix_s1])
            S_matrix_s2 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s2')([SL_matrix,SR_matrix_s2])
            S_matrix_s3 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s3')([SL_matrix,SR_matrix_s3])

            # Very important!!! Without this training won't converge.
            # norm_S = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix)
            norm_S_s1 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s1)
            norm_S_s2 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s2)
            norm_S_s3 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s3)

            feat_s1_pre = Reshape((-1,64))(input_s1_preS)
            feat_s2_pre = Reshape((-1,64))(input_s2_preS)
            feat_s3_pre = Reshape((-1,64))(input_s3_preS)
            feat_pre_concat = Concatenate(axis=1)([feat_s1_pre, feat_s2_pre, feat_s3_pre])
            
            # Warining: don't use keras's 'K.dot'. It is very weird when high dimension is used.
            # https://github.com/keras-team/keras/issues/9779
            # Make sure 'tf.matmul' is used
            # primcaps = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix,feat_pre_concat, norm_S])
            primcaps_s1 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s1,feat_pre_concat, norm_S_s1])
            primcaps_s2 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s2,feat_pre_concat, norm_S_s2])
            primcaps_s3 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s3,feat_pre_concat, norm_S_s3])
            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build(self.num_primcaps,self.m_dim)        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Agg_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            #capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)
            agg_feat = lpk.NetVLAD(feature_size=64, max_samples=self.num_primcaps, cluster_size=self.num_capsule, output_dim=self.num_capsule*self.dim_capsule)(input_primcaps)
            agg_feat = Reshape((self.num_capsule,self.dim_capsule))(agg_feat)

            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(agg_feat)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(agg_feat)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(agg_feat)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Agg_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Agg_model')            
            return ssr_Agg_model

        ssr_Agg_model = ssr_Agg_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Dense(2*self.num_classes,activation='tanh')(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Dense(2*self.num_classes,activation='tanh')(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(input_s1_pre) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Dense(2*self.num_classes,activation='tanh')(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Dense(2*self.num_classes,activation='tanh')(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(input_s2_pre) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Dense(2*self.num_classes,activation='tanh')(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Dense(2*self.num_classes,activation='tanh')(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(input_s3_pre) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
       
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        ssr_F_model = ssr_F_model_build(self.F_shape,'ssr_F_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Agg_list = ssr_Agg_model(ssr_primcaps)
        ssr_F_list = ssr_F_model(ssr_Agg_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_list)
        

        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model
class FSA_net_noS_NetVLAD_FC:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

            
    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build():
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            primcaps_s1 = Reshape((8*8,64))(input_s1_preS)
            primcaps_s2 = Reshape((8*8,64))(input_s2_preS)
            primcaps_s3 = Reshape((8*8,64))(input_s3_preS)
            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build()        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Agg_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            #capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)
            agg_feat = lpk.NetVLAD(feature_size=64, max_samples=self.num_primcaps, cluster_size=self.num_capsule, output_dim=self.num_capsule*self.dim_capsule)(input_primcaps)
            agg_feat = Reshape((self.num_capsule,self.dim_capsule))(agg_feat)

            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(agg_feat)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(agg_feat)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(agg_feat)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Agg_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Agg_model')            
            return ssr_Agg_model

        ssr_Agg_model = ssr_Agg_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Dense(2*self.num_classes,activation='tanh')(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Dense(2*self.num_classes,activation='tanh')(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(input_s1_pre) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Dense(2*self.num_classes,activation='tanh')(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Dense(2*self.num_classes,activation='tanh')(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(input_s2_pre) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Dense(2*self.num_classes,activation='tanh')(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Dense(2*self.num_classes,activation='tanh')(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(input_s3_pre) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
       
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        ssr_F_model = ssr_F_model_build(self.F_shape,'ssr_F_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Agg_list = ssr_Agg_model(ssr_primcaps)
        ssr_F_list = ssr_F_model(ssr_Agg_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_list)
        

        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model


class FSA_net_Metric:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

        
    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        def ssr_feat_S_model_build(num_primcaps, m_dim):
            input_preS = Input((8,8,64))

            feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
            feat_preS = Reshape((-1,))(feat_preS)
            SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
            SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
            
            ssr_feat_S_model = Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')
            return ssr_feat_S_model

        ssr_feat_S_model = ssr_feat_S_model_build(self.num_primcaps,self.m_dim)  
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build(num_primcaps, m_dim):
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            SR_matrix_s1,feat_s1_preS = ssr_feat_S_model(input_s1_preS)
            SR_matrix_s2,feat_s2_preS = ssr_feat_S_model(input_s2_preS)
            SR_matrix_s3,feat_s3_preS = ssr_feat_S_model(input_s3_preS)
            
            feat_pre_concat = Concatenate()([feat_s1_preS,feat_s2_preS,feat_s3_preS])
            SL_matrix = Dense(int(num_primcaps/3)*m_dim,activation='sigmoid')(feat_pre_concat)
            SL_matrix = Reshape((int(num_primcaps/3),m_dim))(SL_matrix)
            
            S_matrix_s1 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s1')([SL_matrix,SR_matrix_s1])
            S_matrix_s2 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s2')([SL_matrix,SR_matrix_s2])
            S_matrix_s3 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s3')([SL_matrix,SR_matrix_s3])

            # Very important!!! Without this training won't converge.
            # norm_S = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix)
            norm_S_s1 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s1)
            norm_S_s2 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s2)
            norm_S_s3 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s3)

            feat_s1_pre = Reshape((-1,64))(input_s1_preS)
            feat_s2_pre = Reshape((-1,64))(input_s2_preS)
            feat_s3_pre = Reshape((-1,64))(input_s3_preS)
            feat_pre_concat = Concatenate(axis=1)([feat_s1_pre, feat_s2_pre, feat_s3_pre])
            
            # Warining: don't use keras's 'K.dot'. It is very weird when high dimension is used.
            # https://github.com/keras-team/keras/issues/9779
            # Make sure 'tf.matmul' is used
            # primcaps = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix,feat_pre_concat, norm_S])
            primcaps_s1 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s1,feat_pre_concat, norm_S_s1])
            primcaps_s2 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s2,feat_pre_concat, norm_S_s2])
            primcaps_s3 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s3,feat_pre_concat, norm_S_s3])
            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build(self.num_primcaps,self.m_dim)        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Metric_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            #capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)
            metric_feat = MatMulLayer(16,type=1)(input_primcaps)
            metric_feat = MatMulLayer(3,type=2)(metric_feat)


            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(metric_feat)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(metric_feat)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(metric_feat)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Metric_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Metric_model')            
            return ssr_Metric_model

        ssr_Metric_model = ssr_Metric_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Lambda(lambda x: x[:,0:4])(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Lambda(lambda x: x[:,4:8])(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Lambda(lambda x: x[:,8:16])(input_s1_pre)
            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(feat_pred_s1) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Lambda(lambda x: x[:,0:4])(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Lambda(lambda x: x[:,4:8])(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Lambda(lambda x: x[:,8:16])(input_s2_pre)
            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(feat_pred_s2) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Lambda(lambda x: x[:,0:4])(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Lambda(lambda x: x[:,4:8])(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Lambda(lambda x: x[:,8:16])(input_s3_pre)
            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(feat_pred_s3) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
        
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        ssr_F_model = ssr_F_model_build(self.F_shape,'ssr_F_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Metric_list = ssr_Metric_model(ssr_primcaps)
        ssr_F_list = ssr_F_model(ssr_Metric_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_list)
        

        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model
class FSA_net_Var_Metric:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

        
    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        def ssr_feat_S_model_build(num_primcaps, m_dim):
            input_preS = Input((8,8,64))

            #feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
            def var(x):
                mean, var = tf.nn.moments(x,axes=-1)
                return var
            feat_preS = Lambda(var)(input_preS)
            feat_preS = Reshape((-1,))(feat_preS)
            SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
            SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
            
            ssr_feat_S_model = Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')
            return ssr_feat_S_model

        ssr_feat_S_model = ssr_feat_S_model_build(self.num_primcaps,self.m_dim)  
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build(num_primcaps, m_dim):
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            SR_matrix_s1,feat_s1_preS = ssr_feat_S_model(input_s1_preS)
            SR_matrix_s2,feat_s2_preS = ssr_feat_S_model(input_s2_preS)
            SR_matrix_s3,feat_s3_preS = ssr_feat_S_model(input_s3_preS)
            
            feat_pre_concat = Concatenate()([feat_s1_preS,feat_s2_preS,feat_s3_preS])
            SL_matrix = Dense(int(num_primcaps/3)*m_dim,activation='sigmoid')(feat_pre_concat)
            SL_matrix = Reshape((int(num_primcaps/3),m_dim))(SL_matrix)
            
            S_matrix_s1 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s1')([SL_matrix,SR_matrix_s1])
            S_matrix_s2 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s2')([SL_matrix,SR_matrix_s2])
            S_matrix_s3 = Lambda(lambda x: tf.matmul(x[0],x[1]),name='S_matrix_s3')([SL_matrix,SR_matrix_s3])

            # Very important!!! Without this training won't converge.
            # norm_S = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix)
            norm_S_s1 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s1)
            norm_S_s2 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s2)
            norm_S_s3 = Lambda(lambda x: K.tile(K.sum(x,axis=-1,keepdims=True),(1,1,64)))(S_matrix_s3)

            feat_s1_pre = Reshape((-1,64))(input_s1_preS)
            feat_s2_pre = Reshape((-1,64))(input_s2_preS)
            feat_s3_pre = Reshape((-1,64))(input_s3_preS)
            feat_pre_concat = Concatenate(axis=1)([feat_s1_pre, feat_s2_pre, feat_s3_pre])
            
            # Warining: don't use keras's 'K.dot'. It is very weird when high dimension is used.
            # https://github.com/keras-team/keras/issues/9779
            # Make sure 'tf.matmul' is used
            # primcaps = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix,feat_pre_concat, norm_S])
            primcaps_s1 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s1,feat_pre_concat, norm_S_s1])
            primcaps_s2 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s2,feat_pre_concat, norm_S_s2])
            primcaps_s3 = Lambda(lambda x: tf.matmul(x[0],x[1])/x[2])([S_matrix_s3,feat_pre_concat, norm_S_s3])
            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build(self.num_primcaps,self.m_dim)        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Metric_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            #capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)
            metric_feat = MatMulLayer(16,type=1)(input_primcaps)
            metric_feat = MatMulLayer(3,type=2)(metric_feat)


            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(metric_feat)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(metric_feat)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(metric_feat)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Metric_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Metric_model')            
            return ssr_Metric_model

        ssr_Metric_model = ssr_Metric_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Lambda(lambda x: x[:,0:4])(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Lambda(lambda x: x[:,4:8])(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Lambda(lambda x: x[:,8:16])(input_s1_pre)
            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(feat_pred_s1) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Lambda(lambda x: x[:,0:4])(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Lambda(lambda x: x[:,4:8])(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Lambda(lambda x: x[:,8:16])(input_s2_pre)
            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(feat_pred_s2) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Lambda(lambda x: x[:,0:4])(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Lambda(lambda x: x[:,4:8])(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Lambda(lambda x: x[:,8:16])(input_s3_pre)
            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(feat_pred_s3) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
        
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        ssr_F_model = ssr_F_model_build(self.F_shape,'ssr_F_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Metric_list = ssr_Metric_model(ssr_primcaps)
        ssr_F_list = ssr_F_model(ssr_Metric_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_list)
        


        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model
class FSA_net_noS_Metric:
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

        self.num_classes = num_classes
        self.stage_num = stage_num
        self.lambda_d = lambda_d

        self.num_capsule = S_set[0]
        self.dim_capsule = S_set[1]
        self.routings = S_set[2]

        self.num_primcaps = S_set[3]
        self.m_dim = S_set[4]

        self.F_shape = int(self.num_capsule/3)*self.dim_capsule

        
    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        def ssr_G_model_build(img_inputs):
            #-------------------------------------------------------------------------------------------------------------------------
            x = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer1 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x_layer1)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(32,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer2 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x_layer2)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(64,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x_layer3 = AveragePooling2D((2,2))(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x_layer3)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128,(3,3),padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x_layer4 = Activation('relu')(x)
            #-------------------------------------------------------------------------------------------------------------------------
            s = SeparableConv2D(16,(3,3),padding='same')(img_inputs)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer1 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s_layer1)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(32,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer2 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s_layer2)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(64,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s_layer3 = MaxPooling2D((2,2))(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s_layer3)
            s = BatchNormalization(axis=-1)(s)
            s = Activation('tanh')(s)
            s = SeparableConv2D(128,(3,3),padding='same')(s)
            s = BatchNormalization(axis=-1)(s)
            s_layer4 = Activation('tanh')(s)
            #-------------------------------------------------------------------------------------------------------------------------
            
            s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
            # s_layer4 = MaxPooling2D((2,2))(s_layer4)

            x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)
            # x_layer4 = AveragePooling2D((2,2))(x_layer4)

            feat_s1_pre = Multiply()([s_layer4,x_layer4])
            # feat_s1_pre = Flatten()(feat_s1_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
            # s_layer3 = MaxPooling2D((2,2))(s_layer3)

            x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)
            # x_layer3 = AveragePooling2D((2,2))(x_layer3)

            feat_s2_pre = Multiply()([s_layer3,x_layer3])
            # feat_s2_pre  = Flatten()(feat_s2_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
            # s_layer2 = MaxPooling2D((2,2))(s_layer2)

            x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)
            # x_layer2 = AveragePooling2D((2,2))(x_layer2)

            feat_s3_pre = Multiply()([s_layer2,x_layer2])
            # feat_s3_pre  = Flatten()(feat_s3_pre)
            #-------------------------------------------------------------------------------------------------------------------------
            
            # Spatial Pyramid Pooling
            #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
            #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
            #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
            # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
            # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
            feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
        
            ssr_G_model = Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')
            return ssr_G_model

        ssr_G_model = ssr_G_model_build(img_inputs)
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_S_model_build():
            input_s1_preS = Input((8,8,64))
            input_s2_preS = Input((8,8,64))
            input_s3_preS = Input((8,8,64))

            primcaps_s1 = Reshape((8*8,64))(input_s1_preS)
            primcaps_s2 = Reshape((8*8,64))(input_s2_preS)
            primcaps_s3 = Reshape((8*8,64))(input_s3_preS)

            primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])

            ssr_S_model = Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')
            return ssr_S_model
        
        ssr_S_model = ssr_S_model_build()        
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_Metric_model_build(shape_primcaps):
            input_primcaps = Input(shape_primcaps)

            #capsule = CapsuleLayer(self.num_capsule, self.dim_capsule, self.routings, name='caps')(input_primcaps)
            metric_feat = MatMulLayer(16,type=1)(input_primcaps)
            metric_feat = MatMulLayer(3,type=2)(metric_feat)


            s1_a = 0
            s1_b = self.num_capsule//3
            feat_s1_div = Lambda(lambda x: x[:,s1_a:s1_b,:])(metric_feat)
            s2_a = self.num_capsule//3
            s2_b = 2*self.num_capsule//3
            feat_s2_div = Lambda(lambda x: x[:,s2_a:s2_b,:])(metric_feat)
            s3_a = 2*self.num_capsule//3
            s3_b = self.num_capsule
            feat_s3_div = Lambda(lambda x: x[:,s3_a:s3_b,:])(metric_feat)


            feat_s1_div = Reshape((-1,))(feat_s1_div)
            feat_s2_div = Reshape((-1,))(feat_s2_div)
            feat_s3_div = Reshape((-1,))(feat_s3_div)
            
            ssr_Metric_model = Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Metric_model')            
            return ssr_Metric_model

        ssr_Metric_model = ssr_Metric_model_build((self.num_primcaps,64))
        #-------------------------------------------------------------------------------------------------------------------------
        def ssr_F_model_build(feat_dim, name_F):
            input_s1_pre = Input((feat_dim,))
            input_s2_pre = Input((feat_dim,))
            input_s3_pre = Input((feat_dim,))

            feat_delta_s1 = Lambda(lambda x: x[:,0:4])(input_s1_pre)
            delta_s1 = Dense(self.num_classes,activation='tanh',name='delta_s1')(feat_delta_s1)

            feat_local_s1 = Lambda(lambda x: x[:,4:8])(input_s1_pre)
            local_s1 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage1')(feat_local_s1)

            feat_pred_s1 = Lambda(lambda x: x[:,8:16])(input_s1_pre)
            feat_pred_s1 = Dense(self.stage_num[0]*self.num_classes,activation='relu')(feat_pred_s1) 
            pred_s1 = Reshape((self.num_classes,self.stage_num[0]))(feat_pred_s1)
            

            feat_delta_s2 = Lambda(lambda x: x[:,0:4])(input_s2_pre)
            delta_s2 = Dense(self.num_classes,activation='tanh',name='delta_s2')(feat_delta_s2)

            feat_local_s2 = Lambda(lambda x: x[:,4:8])(input_s2_pre)
            local_s2 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage2')(feat_local_s2)

            feat_pred_s2 = Lambda(lambda x: x[:,8:16])(input_s2_pre)
            feat_pred_s2 = Dense(self.stage_num[1]*self.num_classes,activation='relu')(feat_pred_s2) 
            pred_s2 = Reshape((self.num_classes,self.stage_num[1]))(feat_pred_s2)
            

            feat_delta_s3 = Lambda(lambda x: x[:,0:4])(input_s3_pre)
            delta_s3 = Dense(self.num_classes,activation='tanh',name='delta_s3')(feat_delta_s3)

            feat_local_s3 = Lambda(lambda x: x[:,4:8])(input_s3_pre)
            local_s3 = Dense(units=self.num_classes, activation='tanh', name='local_delta_stage3')(feat_local_s3)

            feat_pred_s3 = Lambda(lambda x: x[:,8:16])(input_s3_pre)
            feat_pred_s3 = Dense(self.stage_num[2]*self.num_classes,activation='relu')(feat_pred_s3) 
            pred_s3 = Reshape((self.num_classes,self.stage_num[2]))(feat_pred_s3)
        
            ssr_F_model = Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)
            return ssr_F_model
        
        ssr_F_model = ssr_F_model_build(self.F_shape,'ssr_F_model')
        #-------------------------------------------------------------------------------------------------------------------------

        def SSR_module(x,s1,s2,s3,lambda_d):
            a = x[0][:,:,0]*0
            b = x[0][:,:,0]*0
            c = x[0][:,:,0]*0

            di = s1//2
            dj = s2//2
            dk = s3//2

            V = 99
            #lambda_d = 0.9

            for i in range(0,s1):
                a = a+(i-di+x[6])*x[0][:,:,i]
            a = a/(s1*(1+lambda_d*x[3]))

            for j in range(0,s2):
                b = b+(j-dj+x[7])*x[1][:,:,j]
            b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

            for k in range(0,s3):
                c = c+(k-dk+x[8])*x[2][:,:,k]
            c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))

            pred = (a+b+c)*V
            
            return pred

        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Metric_list = ssr_Metric_model(ssr_primcaps)
        ssr_F_list = ssr_F_model(ssr_Metric_list)
        pred_pose = Lambda(SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_list)
        


        model = Model(inputs=img_inputs, outputs=pred_pose)


        return model



