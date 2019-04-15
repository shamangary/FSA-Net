import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np




def get_args():
    parser = argparse.ArgumentParser(description="This script shows training graph from history file.")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input history h5 file")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_path = args.input

    df = pd.read_hdf(input_path, "history")
    print(df[30:60])

    print(np.min(df['val_loss']))
    # input_dir = os.path.dirname(input_path)
    # plt.plot(df["loss"], '-o', label="loss (age)", linewidth=2.0)
    # plt.plot(df["val_loss"], '-o', label="val_loss (age)", linewidth=2.0)
    # plt.xlabel("Number of epochs", fontsize=20)
    # plt.ylabel("Loss", fontsize=20)
    # plt.legend()
    # plt.grid()
    # plt.savefig(os.path.join(input_dir, "loss.pdf"), bbox_inches='tight', pad_inches=0)
    # plt.cla()

    # plt.plot(df["mean_absolute_error"], '-o', label="training", linewidth=2.0)
    # plt.plot(df["val_mean_absolute_error"], '-o', label="validation", linewidth=2.0)
    # ax = plt.gca()
    # ax.set_ylim([2,13])
    # ax.set_aspect(0.6/ax.get_data_ratio())
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.xlabel("Number of epochs", fontsize=20)
    # plt.ylabel("Mean absolute error", fontsize=20)
    # plt.legend(fontsize=20)
    # plt.grid()
    # plt.savefig(os.path.join(input_dir, "performance.pdf"), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()










