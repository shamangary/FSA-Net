import sys
import logging
import numpy as np

import tensorflow as tf

from keras.models import Model

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import Multiply
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization

from keras import backend as K

from .capsulelayers import CapsuleLayer
from .capsulelayers import MatMulLayer

from .loupe_keras import NetVLAD

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)

class BaseFSANet(object):
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

        self.is_fc_model = False
        self.is_noS_model = False

    def _convBlock(self, x, num_filters, activation, kernel_size=(3,3)):
        x = SeparableConv2D(num_filters,kernel_size,padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation(activation)(x)
        return x


    def ssr_G_model_build(self, img_inputs):
        #-------------------------------------------------------------------------------------------------------------------------        
        x = self._convBlock(img_inputs, num_filters=16, activation='relu')
        x_layer1 = AveragePooling2D((2,2))(x)
        x = self._convBlock(x_layer1, num_filters=32, activation='relu')
        x = self._convBlock(x, num_filters=32, activation='relu')        
        x_layer2 = AveragePooling2D((2,2))(x)
        x = self._convBlock(x_layer2, num_filters=64, activation='relu')
        x = self._convBlock(x, num_filters=64, activation='relu')        
        x_layer3 = AveragePooling2D((2,2))(x)
        x = self._convBlock(x_layer3, num_filters=128, activation='relu')
        x_layer4 = self._convBlock(x, num_filters=128, activation='relu')                        
        #-------------------------------------------------------------------------------------------------------------------------
        s = self._convBlock(img_inputs, num_filters=16, activation='tanh')
        s_layer1 = MaxPooling2D((2,2))(s)
        s = self._convBlock(s_layer1, num_filters=32, activation='tanh')
        s = self._convBlock(s, num_filters=32, activation='tanh')        
        s_layer2 = MaxPooling2D((2,2))(s)
        s = self._convBlock(s_layer2, num_filters=64, activation='tanh')
        s = self._convBlock(s, num_filters=64, activation='tanh')        
        s_layer3 = MaxPooling2D((2,2))(s)
        s = self._convBlock(s_layer3, num_filters=128, activation='tanh')
        s_layer4 = self._convBlock(s, num_filters=128, activation='tanh')                                
        #-------------------------------------------------------------------------------------------------------------------------        
        s_layer4 = Conv2D(64,(1,1),activation='tanh')(s_layer4)
        x_layer4 = Conv2D(64,(1,1),activation='relu')(x_layer4)

        feat_s1_pre = Multiply()([s_layer4,x_layer4])
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer3 = Conv2D(64,(1,1),activation='tanh')(s_layer3)
        x_layer3 = Conv2D(64,(1,1),activation='relu')(x_layer3)

        feat_s2_pre = Multiply()([s_layer3,x_layer3])
        #-------------------------------------------------------------------------------------------------------------------------
        s_layer2 = Conv2D(64,(1,1),activation='tanh')(s_layer2)
        x_layer2 = Conv2D(64,(1,1),activation='relu')(x_layer2)

        feat_s3_pre = Multiply()([s_layer2,x_layer2])
        #-------------------------------------------------------------------------------------------------------------------------
        
        # Spatial Pyramid Pooling
        #feat_s1_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s1_pre)        
        #feat_s2_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s2_pre)
        #feat_s3_pre = SpatialPyramidPooling([1, 2, 4],'average')(feat_s3_pre)
        # feat_s1_pre = GlobalAveragePooling2D()(feat_s1_pre)
        # feat_s2_pre = GlobalAveragePooling2D()(feat_s2_pre)
        feat_s3_pre = AveragePooling2D((2,2))(feat_s3_pre) # make sure (8x8x64) feature maps 
    
        return Model(inputs=img_inputs,outputs=[feat_s1_pre,feat_s2_pre,feat_s3_pre], name='ssr_G_model')

    def SSR_module(self, x,s1,s2,s3,lambda_d):
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

    def ssr_F_model_build(self, feat_dim, name_F):
        input_s1_pre = Input((feat_dim,))
        input_s2_pre = Input((feat_dim,))
        input_s3_pre = Input((feat_dim,))

        def _process_input(stage_index, stage_num, num_classes, input_s_pre):
            feat_delta_s = Lambda(lambda x: x[:,0:4])(input_s_pre)
            delta_s = Dense(num_classes,activation='tanh',name=f'delta_s{stage_index}')(feat_delta_s)

            feat_local_s = Lambda(lambda x: x[:,4:8])(input_s_pre)
            local_s = Dense(units=num_classes, activation='tanh', name=f'local_delta_stage{stage_index}')(feat_local_s)

            feat_pred_s = Lambda(lambda x: x[:,8:16])(input_s_pre)
            feat_pred_s = Dense(stage_num*num_classes,activation='relu')(feat_pred_s) 
            pred_s = Reshape((num_classes,stage_num))(feat_pred_s)

            return delta_s, local_s, pred_s

        delta_s1, local_s1, pred_s1 = _process_input(1, self.stage_num[0], self.num_classes, input_s1_pre)
        delta_s2, local_s2, pred_s2 = _process_input(2, self.stage_num[1], self.num_classes, input_s2_pre)
        delta_s3, local_s3, pred_s3 = _process_input(3, self.stage_num[2], self.num_classes, input_s3_pre)        
    
        return Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)

    def ssr_FC_model_build(self, feat_dim, name_F):
        input_s1_pre = Input((feat_dim,))
        input_s2_pre = Input((feat_dim,))
        input_s3_pre = Input((feat_dim,))

        def _process_input(stage_index, stage_num, num_classes, input_s_pre):
            feat_delta_s = Dense(2*num_classes,activation='tanh')(input_s_pre)
            delta_s = Dense(num_classes,activation='tanh',name=f'delta_s{stage_index}')(feat_delta_s)

            feat_local_s = Dense(2*num_classes,activation='tanh')(input_s_pre)
            local_s = Dense(units=num_classes, activation='tanh', name=f'local_delta_stage{stage_index}')(feat_local_s)

            feat_pred_s = Dense(stage_num*num_classes,activation='relu')(input_s_pre) 
            pred_s = Reshape((num_classes,stage_num))(feat_pred_s)     

            return delta_s, local_s, pred_s   

        delta_s1, local_s1, pred_s1 = _process_input(1, self.stage_num[0], self.num_classes, input_s1_pre)
        delta_s2, local_s2, pred_s2 = _process_input(2, self.stage_num[1], self.num_classes, input_s2_pre)
        delta_s3, local_s3, pred_s3 = _process_input(3, self.stage_num[2], self.num_classes, input_s3_pre)        
           
        return Model(inputs=[input_s1_pre,input_s2_pre,input_s3_pre],outputs=[pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3,local_s1,local_s2,local_s3], name=name_F)

    def ssr_S_model_build(self, num_primcaps, m_dim):
        input_s1_preS = Input((8,8,64))
        input_s2_preS = Input((8,8,64))
        input_s3_preS = Input((8,8,64))

        feat_S_model = self.ssr_feat_S_model_build(m_dim)

        SR_matrix_s1,feat_s1_preS = feat_S_model(input_s1_preS)
        SR_matrix_s2,feat_s2_preS = feat_S_model(input_s2_preS)
        SR_matrix_s3,feat_s3_preS = feat_S_model(input_s3_preS)
        
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

        return Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')

    def ssr_noS_model_build(self, **kwargs):
        input_s1_preS = Input((8,8,64))
        input_s2_preS = Input((8,8,64))
        input_s3_preS = Input((8,8,64))

        primcaps_s1 = Reshape((8*8,64))(input_s1_preS)
        primcaps_s2 = Reshape((8*8,64))(input_s2_preS)
        primcaps_s3 = Reshape((8*8,64))(input_s3_preS)

        primcaps = Concatenate(axis=1)([primcaps_s1,primcaps_s2,primcaps_s3])
        return Model(inputs=[input_s1_preS, input_s2_preS, input_s3_preS],outputs=primcaps, name='ssr_S_model')


# Capsule FSANetworks

class BaseCapsuleFSANet(BaseFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(BaseCapsuleFSANet, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)         

    def ssr_Cap_model_build(self, shape_primcaps):
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
        
        return Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Cap_model') 
    
    def __call__(self):
        logging.debug("Creating model...")
        img_inputs = Input(self._input_shape)        

        # Build various models
        ssr_G_model = self.ssr_G_model_build(img_inputs)        
        
        if self.is_noS_model:
            ssr_S_model = self.ssr_noS_model_build()           
        else:
            ssr_S_model = self.ssr_S_model_build(num_primcaps=self.num_primcaps,m_dim=self.m_dim)           

        ssr_Cap_model = self.ssr_Cap_model_build((self.num_primcaps,64))

        if self.is_fc_model:
            ssr_F_Cap_model = self.ssr_FC_model_build(self.F_shape,'ssr_F_Cap_model')
        else:    
            ssr_F_Cap_model = self.ssr_F_model_build(self.F_shape,'ssr_F_Cap_model')

        # Wire them up
        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Cap_list = ssr_Cap_model(ssr_primcaps)
        ssr_F_Cap_list = ssr_F_Cap_model(ssr_Cap_list)
        pred_pose = Lambda(self.SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_Cap_list)
        
        return Model(inputs=img_inputs, outputs=pred_pose)

class FSA_net_Capsule(BaseCapsuleFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Capsule, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)       

    def ssr_feat_S_model_build(self, num_primcaps, m_dim):
        input_preS = Input((8,8,64))
        feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
        feat_preS = Reshape((-1,))(feat_preS)
        SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
        SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
        
        return Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')    
 
class FSA_net_Var_Capsule(BaseCapsuleFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Var_Capsule, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)   

    def ssr_feat_S_model_build(self, m_dim):
        input_preS = Input((8,8,64))
        def var(x):
            mean, var = tf.nn.moments(x,axes=-1)
            return var
        feat_preS = Lambda(var)(input_preS)
        feat_preS = Reshape((-1,))(feat_preS)
        SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
        SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
        
        return Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')
        
class FSA_net_noS_Capsule(BaseCapsuleFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_noS_Capsule, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)   
        self.is_noS_model = True    
    
class FSA_net_Capsule_FC(FSA_net_Capsule):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Capsule_FC, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)  
        self.is_fc_model = True

class FSA_net_Var_Capsule_FC(FSA_net_Var_Capsule):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Var_Capsule_FC, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)
        self.is_fc_model = True
        
class FSA_net_noS_Capsule_FC(FSA_net_noS_Capsule):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_noS_Capsule_FC, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)
        self.is_fc_model = True
   
# NetVLAD models

class BaseNetVLADFSANet(BaseFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(BaseNetVLADFSANet, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)         

    def ssr_Agg_model_build(self, shape_primcaps):
        input_primcaps = Input(shape_primcaps)
        
        agg_feat = NetVLAD(feature_size=64, max_samples=self.num_primcaps, cluster_size=self.num_capsule, output_dim=self.num_capsule*self.dim_capsule)(input_primcaps)
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
        
        return Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Agg_model')      

    def __call__(self):
        logging.debug("Creating model...")

        img_inputs = Input(self._input_shape)
        ssr_G_model = self.ssr_G_model_build(img_inputs)        
        
        if self.is_noS_model:
            ssr_S_model = self.ssr_noS_model_build()           
        else:
            ssr_S_model = self.ssr_S_model_build(num_primcaps=self.num_primcaps,m_dim=self.m_dim)           

        ssr_Agg_model = self.ssr_Agg_model_build((self.num_primcaps,64))

        if self.is_fc_model:
            ssr_F_model  =self.ssr_FC_model_build(self.F_shape,'ssr_F_model')
        else:
            ssr_F_model = self.ssr_F_model_build(self.F_shape,'ssr_F_model')
        
        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Agg_list = ssr_Agg_model(ssr_primcaps)
        ssr_F_list = ssr_F_model(ssr_Agg_list)
        pred_pose = Lambda(self.SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_list)

        return Model(inputs=img_inputs, outputs=pred_pose)          

class FSA_net_NetVLAD(BaseNetVLADFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_NetVLAD, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)

    def ssr_feat_S_model_build(self, m_dim):
        input_preS = Input((8,8,64))

        feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
        feat_preS = Reshape((-1,))(feat_preS)
        SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
        SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
        
        return Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')    
    
class FSA_net_Var_NetVLAD(BaseNetVLADFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Var_NetVLAD, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)

    def ssr_feat_S_model_build(self, m_dim):
        input_preS = Input((8,8,64))

        #feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
        def var(x):
            mean, var = tf.nn.moments(x,axes=-1)
            return var
        feat_preS = Lambda(var)(input_preS)
        feat_preS = Reshape((-1,))(feat_preS)
        SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
        SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
        
        return Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')   

class FSA_net_noS_NetVLAD(BaseNetVLADFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_noS_NetVLAD, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)
        self.is_noS_model = True

class FSA_net_NetVLAD_FC(FSA_net_NetVLAD):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_NetVLAD_FC, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)
        self.is_fc_model = True

class FSA_net_Var_NetVLAD_FC(FSA_net_Var_NetVLAD):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Var_NetVLAD_FC, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)
        self.is_fc_model = True

class FSA_net_noS_NetVLAD_FC(FSA_net_noS_NetVLAD):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_noS_NetVLAD_FC, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)
        self.is_fc_model = True

# // Metric models

class BaseMetricFSANet(BaseFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(BaseMetricFSANet, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set) 
        
    def ssr_Metric_model_build(self, shape_primcaps):
        input_primcaps = Input(shape_primcaps)

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
        
        return Model(inputs=input_primcaps,outputs=[feat_s1_div,feat_s2_div,feat_s3_div], name='ssr_Metric_model')     

    def __call__(self):
        logging.debug("Creating model...")
        
        # build models
        img_inputs = Input(self._input_shape)
        ssr_G_model = self.ssr_G_model_build(img_inputs)
        
        if self.is_noS_model:
            ssr_S_model = self.ssr_noS_model_build()           
        else:
            ssr_S_model = self.ssr_S_model_build(num_primcaps=self.num_primcaps,m_dim=self.m_dim)           

        ssr_Metric_model = self.ssr_Metric_model_build((self.num_primcaps,64))
        ssr_F_model = self.ssr_F_model_build(self.F_shape,'ssr_F_model')
        
        # wire models
        ssr_G_list = ssr_G_model(img_inputs)
        ssr_primcaps = ssr_S_model(ssr_G_list)
        ssr_Metric_list = ssr_Metric_model(ssr_primcaps)
        ssr_F_list = ssr_F_model(ssr_Metric_list)
        pred_pose = Lambda(self.SSR_module,arguments={'s1':self.stage_num[0],'s2':self.stage_num[1],'s3':self.stage_num[2],'lambda_d':self.lambda_d},name='pred_pose')(ssr_F_list)
       
        return Model(inputs=img_inputs, outputs=pred_pose)

class FSA_net_Metric(BaseMetricFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Metric, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)

    def ssr_feat_S_model_build(self, m_dim):
        input_preS = Input((8,8,64))

        feat_preS = Conv2D(1,(1,1),padding='same',activation='sigmoid')(input_preS)
        feat_preS = Reshape((-1,))(feat_preS)
        SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
        SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
        
        return Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')    

class FSA_net_Var_Metric(BaseMetricFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_Var_Metric, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)

    def ssr_feat_S_model_build(self, m_dim):
        input_preS = Input((8,8,64))
        def var(x):
            mean, var = tf.nn.moments(x,axes=-1)
            return var
        feat_preS = Lambda(var)(input_preS)
        feat_preS = Reshape((-1,))(feat_preS)
        SR_matrix = Dense(m_dim*(8*8+8*8+8*8),activation='sigmoid')(feat_preS)
        SR_matrix = Reshape((m_dim,(8*8+8*8+8*8)))(SR_matrix)
        
        return Model(inputs=input_preS,outputs=[SR_matrix,feat_preS],name='feat_S_model')    
        
class FSA_net_noS_Metric(BaseMetricFSANet):
    def __init__(self, image_size,num_classes,stage_num,lambda_d, S_set):
        super(FSA_net_noS_Metric, self).__init__(image_size,num_classes,stage_num,lambda_d, S_set)
        self.is_noS_model = True
   