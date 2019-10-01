import re
import os
import sys
import importlib
import argparse
sys.path.append('..')

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from lib.FSANET_model import *

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def build_model_class_from_name(model_name):

    is_var_model = "var" in model_name
    is_noS_model = "noS" in model_name
    is_capsule_model = "capsule" in model_name
    is_netvlad_model = "netvlad" in model_name
    is_metric_model = "metric" in model_name

    model_class_name = f"FSA_net_"
    if is_capsule_model:
        if is_var_model:
            model_class_name = model_class_name + "Var_Capsule"
        elif is_noS_model:
            model_class_name = model_class_name + "noS_Capsule"
        else:
            model_class_name = model_class_name + "Capsule"

    if is_netvlad_model:
        if is_var_model:
            model_class_name = model_class_name + "Var_NetVLAD"
        elif is_noS_model:
            model_class_name = model_class_name + "noS_NetVLAD"
        else:
            model_class_name = model_class_name + "NetVLAD"

    if is_metric_model:
        if is_var_model:
            model_class_name = model_class_name + "Var_Metric"
        elif is_noS_model:
            model_class_name = model_class_name + "noS_Metric"
        else:
            model_class_name = model_class_name + "Metric"

    return model_class_name


def create_model(model_name, model_class_name):
    """ Since the archived models have lambda layers in them
    we need to first load them using the new model definitions.
    The load will be successful as I have preserved the names of the trainable layers.
    After that we do need to save it so that it has custom layers and then we load it back again """

    # we will use the name of the directory to dynamically create the corresponding
    # class and the hyper-parameters (which thankfully are encoded in the name of the directory)

    model_cls = class_for_name("lib", model_class_name)

    hparams_start_loc = re.search("\d", model_name).start()
    hprams_str = model_name[hparams_start_loc:]

    # now split it by '_' and cast it as int
    hparams = [int(d) for d in hprams_str.split('_')]

    # static params
    stage_num = [3, 3, 3]
    lambda_d = 1
    num_classes = 3
    image_size = 64

    # now create the model object
    model_obj = model_cls(image_size, num_classes,
                          stage_num, lambda_d, hparams)()

    model_obj.count_params()
    model_obj.summary()

    return model_obj


def parse_arguments(argv):
    """ Parse the arguments """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--trained-model-dir-path',
        required=True,
        type=str,
        help='The directory that contains the pre-trained model')

    parser.add_argument(
        '--output-dir-path',
        required=True,
        type=str,
        help='The directory that would contain the converted models')

    return parser.parse_args(argv)


def main(args):

    model_name = os.path.basename(args.trained_model_dir_path)

    # convert existing model with lambda layers to
    # new model with custom layers
    model_cls_name = build_model_class_from_name(model_name)

    model_obj = create_model(model_name, model_cls_name)
    # need to load the weights first
    model_obj.load_weights(os.path.join(
        args.trained_model_dir_path, model_name + ".h5"))

    # we now save it in the temp folder
    # this version will now be saved with the custom layer information
    # serialized
    converted_keras_model_dir = os.path.join(args.output_dir_path,
                                             "converted-models", "keras")

    os.makedirs(converted_keras_model_dir, exist_ok=True)

    keras_model_path = os.path.join(
        converted_keras_model_dir, model_name) + ".hd5"

    model_obj.save(keras_model_path)

    # Do the session clearing and creation first
    K.clear_session()
    sess = K.get_session()
    K.set_learning_phase(0)

    # Load it back
    model = load_model(keras_model_path)

    print(f"Model inputs information - {model.inputs}")
    print(f"Model outputs information - {model.outputs}")

    # freez the graph
    frozen_graph = freeze_session(sess,
                                  output_names=[out.op.name for out in model.outputs])

    tf_dir_path = os.path.join(args.output_dir_path, "converted-models", "tf")
    os.makedirs(tf_dir_path, exist_ok=True)

    # write the graph
    tf.io.write_graph(frozen_graph, tf_dir_path,
                      f"{model_name}.pb", as_text=False)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))