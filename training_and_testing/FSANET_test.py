import os
import sys
sys.path.append('..')
import logging
import argparse

import numpy as np
import pandas as pd

from keras import backend as K
from keras.layers import *
from keras.utils import plot_model
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from lib.FSANET_model import *
from lib.SSRNET_model import *

import TYY_callbacks
from TYY_generators import *

_TRAIN_DB_300W_LP = "300W_LP"
_TRAIN_DB_BIWI = "BIWI"

_TEST_DB_AFLW = "AFLW2000"
_TEST_DB_BIWI = "BIWI"

_IMAGE_SIZE = 64

def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["pose"]

def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass

def get_weights_file_path(use_pretrained, train_db_name, save_name):
    prefix = "../pre-trained/" if use_pretrained else ""
    return prefix + train_db_name+"_models/"+save_name+"/"+save_name+".h5"    

def get_args():
    parser = argparse.ArgumentParser(description="This script tests the CNN model for head pose estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_type", type=int, default=3, 
                        help="type of model")
    parser.add_argument('--use_pretrained', required=False,
                        dest='use_pretrained',
                        action='store_true')
    parser.add_argument("--train_db", choices=[_TRAIN_DB_300W_LP, _TRAIN_DB_BIWI], required=False, default=_TRAIN_DB_300W_LP)

    parser.set_defaults(use_pretrained=False)

    args = parser.parse_args()
    return args

def main():
    K.clear_session()
    K.set_learning_phase(0) # make sure its testing mode
    
    args = get_args()
    
    model_type = args.model_type
    train_db_name = args.train_db
    use_pretrained = args.use_pretrained   
    

    if train_db_name == _TRAIN_DB_300W_LP:
        test_db_list = [_TEST_DB_AFLW, _TEST_DB_BIWI]
    elif train_db_name == _TRAIN_DB_BIWI:
        test_db_list = [_TEST_DB_BIWI]

    for test_db_name in test_db_list:

        if test_db_name == _TEST_DB_AFLW:            
            image, pose = load_data_npz('../data/type1/AFLW2000.npz')
        elif test_db_name == _TEST_DB_BIWI:            
            if train_db_name == _TRAIN_DB_300W_LP:
                image, pose = load_data_npz('../data/BIWI_noTrack.npz')
            elif train_db_name == _TRAIN_DB_BIWI:
                image, pose = load_data_npz('../data/BIWI_test.npz')
        
        if train_db_name == _TRAIN_DB_300W_LP:
            # we only care the angle between [-99,99] and filter other angles
            x_data = []
            y_data = []

            for i in range(0,pose.shape[0]):
                temp_pose = pose[i,:]
                if np.max(temp_pose)<=99.0 and np.min(temp_pose)>=-99.0:
                    x_data.append(image[i,:,:,:])
                    y_data.append(pose[i,:])
            x_data = np.array(x_data)
            y_data = np.array(y_data)
        else:
            x_data = image
            y_data = pose
        
        stage_num = [3,3,3]
        lambda_d = 1
        num_classes = 3

        if model_type == 0:
            model = SSR_net_ori_MT(_IMAGE_SIZE, num_classes, stage_num, lambda_d)()
            save_name = 'ssrnet_ori_mt'

        elif model_type == 1:
            model = SSR_net_MT(_IMAGE_SIZE, num_classes, stage_num, lambda_d)()
            save_name = 'ssrnet_mt'

        elif model_type == 2:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_capsule'+str_S_set
        
        elif model_type == 3:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Var_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_var_capsule'+str_S_set
        elif model_type == 4:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_NetVLAD(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_netvlad'+str_S_set
        elif model_type == 5:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Var_NetVLAD(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_var_netvlad'+str_S_set
        elif model_type == 6:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_noS_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_noS_capsule'+str_S_set
        elif model_type == 7:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_noS_NetVLAD(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_noS_netvlad'+str_S_set
        
        elif model_type == 8:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_capsule_fine'+str_S_set
            
        elif model_type == 9:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Capsule_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_capsule_fc'+str_S_set
        elif model_type == 10:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Var_Capsule_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_var_capsule_fc'+str_S_set
        elif model_type == 11:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_noS_Capsule_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_noS_capsule_fc'+str_S_set
        elif model_type == 12:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_NetVLAD_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_netvlad_fc'+str_S_set
        elif model_type == 13:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_Var_NetVLAD_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_var_netvlad_fc'+str_S_set
        elif model_type == 14:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model = FSA_net_noS_NetVLAD_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name = 'fsanet_noS_netvlad_fc'+str_S_set

        elif model_type == 15:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model1 = FSA_net_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name1 = 'fsanet_capsule'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model2 = FSA_net_Var_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name2 = 'fsanet_var_capsule'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model3 = FSA_net_noS_Capsule(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name3 = 'fsanet_noS_capsule'+str_S_set
            save_name = 'fusion_dim_split_capsule'
        elif model_type == 16:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model1 = FSA_net_Capsule_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name1 = 'fsanet_capsule_fc'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model2 = FSA_net_Var_Capsule_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name2 = 'fsanet_var_capsule_fc'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model3 = FSA_net_noS_Capsule_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name3 = 'fsanet_noS_capsule_fc'+str_S_set

            save_name = 'fusion_fc_capsule'
        elif model_type == 17:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model1 = FSA_net_NetVLAD(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name1 = 'fsanet_netvlad'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model2 = FSA_net_Var_NetVLAD(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name2 = 'fsanet_var_netvlad'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model3 = FSA_net_noS_NetVLAD(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name3 = 'fsanet_noS_netvlad'+str_S_set

            save_name = 'fusion_dim_split_netvlad'
        elif model_type == 18:
            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model1 = FSA_net_NetVLAD_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name1 = 'fsanet_netvlad_fc'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 7*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model2 = FSA_net_Var_NetVLAD_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name2 = 'fsanet_var_netvlad_fc'+str_S_set

            num_capsule = 3
            dim_capsule = 16
            routings = 2

            num_primcaps = 8*8*3
            m_dim = 5
            S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
            str_S_set = ''.join('_'+str(x) for x in S_set)

            model3 = FSA_net_noS_NetVLAD_FC(_IMAGE_SIZE, num_classes, stage_num, lambda_d, S_set)()
            save_name3 = 'fsanet_noS_netvlad_fc'+str_S_set
            save_name = 'fusion_fc_netvlad'

        if model_type <15:            
            weight_file = get_weights_file_path(use_pretrained, train_db_name, save_name)
            model.load_weights(weight_file)
        else:            
            weight_file1 = get_weights_file_path(use_pretrained, train_db_name, save_name1)
            model1.load_weights(weight_file1)            
            weight_file2 = get_weights_file_path(use_pretrained, train_db_name, save_name2)
            model2.load_weights(weight_file2)            
            weight_file3 = get_weights_file_path(use_pretrained, train_db_name, save_name3)
            model3.load_weights(weight_file3)
            inputs = Input(shape=(64,64,3))
            x1 = model1(inputs)
            x2 = model2(inputs)
            x3 = model3(inputs)
            outputs = Average()([x1,x2,x3])
            model = Model(inputs=inputs,outputs=outputs)

        p_data = model.predict(x_data)
        pose_matrix = np.mean(np.abs(p_data-y_data),axis=0)
        MAE = np.mean(pose_matrix)
        yaw = pose_matrix[0]
        pitch = pose_matrix[1]
        roll = pose_matrix[2]
        print('\n--------------------------------------------------------------------------------')
        print(save_name+', '+test_db_name+'('+train_db_name+')'+', MAE = %3.3f, [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]'%(MAE, yaw, pitch, roll))
        print('--------------------------------------------------------------------------------')

if __name__ == '__main__':    
    main()
