import os
import sys
sys.path.append('..')
import logging
import argparse
import pandas as pd

import numpy as np

from keras.utils import np_utils
from keras.utils import plot_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from lib.FSANET_model import *

import TYY_callbacks
from TYY_generators import *

import cv2
logging.basicConfig(level=logging.DEBUG)


def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["pose"]


def mk_dir(dir):
    try:
        os.mkdir(dir)
    except OSError:
        pass


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for head pose estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=90,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")
    parser.add_argument("--model_type", type=int, default=3,
                        help="type of model")
    parser.add_argument("--db_name", type=str, default='300W_LP',
                        help="type of model")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    db_name = args.db_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split
    model_type = args.model_type
    image_size = 64

    logging.debug("Loading data...")

    if db_name == '300W_LP':
        db_list = ['AFW.npz', 'AFW_Flip.npz', 'HELEN.npz', 'HELEN_Flip.npz',
                   'IBUG.npz', 'IBUG_Flip.npz', 'LFPW.npz', 'LFPW_Flip.npz']
        image = []
        pose = []
        for i in range(0, len(db_list)):
            image_temp, pose_temp = load_data_npz('../data/type1/'+db_list[i])
            image.append(image_temp)
            pose.append(pose_temp)
        image = np.concatenate(image, 0)
        pose = np.concatenate(pose, 0)

        # we only care the angle between [-99,99] and filter other angles
        x_data = []
        y_data = []
        print(image.shape)
        print(pose.shape)
        for i in range(0, pose.shape[0]):
            temp_pose = pose[i, :]
            if np.max(temp_pose) <= 99.0 and np.min(temp_pose) >= -99.0:
                x_data.append(image[i, :, :, :])
                y_data.append(pose[i, :])
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        print(x_data.shape)
        print(y_data.shape)
    elif db_name == 'synhead_noBIWI':
        image, pose = load_data_npz(
            '../data/synhead/media/jinweig/Data2/synhead2_release/synhead_noBIWI.npz')
        x_data = image
        y_data = pose
    elif db_name == 'BIWI':
        image, pose = load_data_npz('../data/BIWI_train.npz')
        x_train = image
        y_train = pose
        image_test, pose_test = load_data_npz('../data/BIWI_test.npz')
        x_test = image_test
        y_test = pose_test
    else:
        print('db_name is wrong!!!')
        return

    start_decay_epoch = [30, 60]

    optMethod = Adam()

    stage_num = [3, 3, 3]
    lambda_d = 1
    num_classes = 3
    isFine = False

    if model_type == 1:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Capsule(image_size, num_classes,
                                stage_num, lambda_d, S_set)()
        load_name = 'fsanet_capsule'+str_S_set
        save_name = 'fsanet_capsule_fine'+str_S_set
        weight_file = "synhead_noBIWI"+"_models/"+load_name+"/"+load_name+".h5"
        model.load_weights(weight_file)
    elif model_type == 2:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Var_Capsule(
            image_size, num_classes, stage_num, lambda_d, S_set)()
        load_name = 'fsanet_var_capsule'+str_S_set
        save_name = 'fsanet_var_capsule_fine'+str_S_set
        weight_file = "synhead_noBIWI"+"_models/"+load_name+"/"+load_name+".h5"
        model.load_weights(weight_file)
    elif model_type == 3:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8*8*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_noS_Capsule(
            image_size, num_classes, stage_num, lambda_d, S_set)()
        load_name = 'fsanet_noS_capsule'+str_S_set
        save_name = 'fsanet_noS_capsule_fine'+str_S_set
        weight_file = "synhead_noBIWI"+"_models/"+load_name+"/"+load_name+".h5"
        model.load_weights(weight_file)

    model.compile(optimizer=optMethod, loss=["mae"], loss_weights=[1])

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    mk_dir(db_name+"_models")
    mk_dir(db_name+"_models/"+save_name)
    mk_dir(db_name+"_checkpoints")
    plot_model(model, to_file=db_name+"_models/" +
               save_name+"/"+save_name+".png")
    for i_L, layer in enumerate(model.layers):
        if i_L > 0 and i_L < len(model.layers)-1:
            if 'pred' not in layer.name and 'caps' != layer.name and 'merge' not in layer.name and 'model' in layer.name:
                plot_model(layer, to_file=db_name+"_models/" +
                           save_name+"/"+layer.name+".png")

    decaylearningrate = TYY_callbacks.DecayLearningRate(start_decay_epoch)

    callbacks = [ModelCheckpoint(db_name+"_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto"), decaylearningrate
                 ]

    logging.debug("Running training...")

    if db_name != 'BIWI':
        data_num = len(x_data)
        indexes = np.arange(data_num)
        np.random.shuffle(indexes)
        x_data = x_data[indexes]
        y_data = y_data[indexes]
        train_num = int(data_num * (1 - validation_split))

        x_train = x_data[:train_num]
        x_test = x_data[train_num:]
        y_train = y_data[:train_num]
        y_test = y_data[train_num:]
    elif db_name == 'BIWI':
        train_num = np.shape(x_train)[0]

    hist = model.fit_generator(generator=data_generator_pose(X=x_train, Y=y_train, batch_size=batch_size),
                               steps_per_epoch=train_num // batch_size,
                               validation_data=(x_test, y_test),
                               epochs=nb_epochs, verbose=1,
                               callbacks=callbacks)

    logging.debug("Saving weights...")
    model.save_weights(os.path.join(db_name+"_models/" +
                                    save_name, save_name+'.h5'), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join(
        db_name+"_models/"+save_name, 'history_'+save_name+'.h5'), "history")


if __name__ == '__main__':
    main()
