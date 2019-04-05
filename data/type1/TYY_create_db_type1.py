import scipy.io as sio
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import sys
import cv2
from moviepy.editor import *
import numpy as np
import argparse


def get_args():
	parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
	                                             "and creates database for training.",
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--db", type=str, default='./AFW',
	                    help="path to database")
	parser.add_argument("--output", type=str, default='./AFW.npz',
	                    help="path to output database mat file")
	parser.add_argument("--img_size", type=int, default=64,
	                    help="output image size")
	parser.add_argument("--ad", type=float, default=0.6,
	                    help="enlarge margin")


	args = parser.parse_args()
	return args


def main():
	args = get_args()
	mypath = args.db
	output_path = args.output
	img_size = args.img_size
	ad = args.ad

	isPlot = False

	onlyfiles_mat = [f for f in listdir(mypath) if isfile(join(mypath, f)) and join(mypath, f).endswith('.mat')]
	onlyfiles_jpg = [f for f in listdir(mypath) if isfile(join(mypath, f)) and join(mypath, f).endswith('.jpg')]
	
	onlyfiles_mat.sort()
	onlyfiles_jpg.sort()
	print(len(onlyfiles_jpg))
	print(len(onlyfiles_mat))
	out_imgs = []
	out_poses = []

	for i in tqdm(range(len(onlyfiles_jpg))):
		img_name = onlyfiles_jpg[i]
		mat_name = onlyfiles_mat[i]

		img_name_split = img_name.split('.')
		mat_name_split = mat_name.split('.')

		if img_name_split[0] != mat_name_split[0]:
			print('Mismatched!')
			sys.exit()

		mat_contents = sio.loadmat(mypath + '/' + mat_name)
		pose_para = mat_contents['Pose_Para'][0]
		pt2d = mat_contents['pt2d']
		
		pt2d_x = pt2d[0,:]
		pt2d_y = pt2d[1,:]

		# I found negative value in AFLW2000. It need to be removed.
		pt2d_idx = pt2d_x>0.0
		pt2d_idy= pt2d_y>0.0

		pt2d_id = pt2d_idx
		if sum(pt2d_idx) > sum(pt2d_idy):
			pt2d_id = pt2d_idy
		
		pt2d_x = pt2d_x[pt2d_id]
		pt2d_y = pt2d_y[pt2d_id]
		
		img = cv2.imread(mypath+'/'+img_name)
		img_h = img.shape[0]
		img_w = img.shape[1]

		# Crop the face loosely
		x_min = int(min(pt2d_x))
		x_max = int(max(pt2d_x))
		y_min = int(min(pt2d_y))
		y_max = int(max(pt2d_y))
		
		h = y_max-y_min
		w = x_max-x_min

		# ad = 0.4
		x_min = max(int(x_min - ad * w), 0)
		x_max = min(int(x_max + ad * w), img_w - 1)
		y_min = max(int(y_min - ad * h), 0)
		y_max = min(int(y_max + ad * h), img_h - 1)
		
		img = img[y_min:y_max,x_min:x_max]
		# Checking the cropped image
		if isPlot:
			cv2.imshow('check',img)
			k=cv2.waitKey(500)

		img = cv2.resize(img, (img_size, img_size))
		
		

		pitch = pose_para[0] * 180 / np.pi
		yaw = pose_para[1] * 180 / np.pi
		roll = pose_para[2] * 180 / np.pi

		cont_labels = np.array([yaw, pitch, roll])

		out_imgs.append(img)
		out_poses.append(cont_labels)

	np.savez(output_path,image=np.array(out_imgs), pose=np.array(out_poses), img_size=img_size)


if __name__ == '__main__':
	main()