import keras
from sklearn.metrics import roc_auc_score
import sys
import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np
from keras import backend as K
import tensorflow as tf


class DecayLearningRate(keras.callbacks.Callback):
	def __init__(self, startEpoch):
		self.startEpoch = startEpoch
		# self.isFine = isFine
	
	# def build_ht_copy_W(self):
	# 	self.ht = {}
	# 	for i_m, sub_model in enumerate(self.model.layers):
	# 		if i_m>0:
	# 			W = sub_model.get_weights() # weight is list
	# 			if type(W) == list:
	# 				self.ht[i_m] = list(W) # copying a list in python
	# 			else:
	# 				sys.exit()

	# 			# If trying to copy a tensor
	# 			# make sure the saved tensor is a copy one
	# 			# (tensor + 0) is necessary

	# 	return
	# def list_operation(self,A_list,B_list,alpha):
	# 	temp = []
	# 	for i_A in range(len(A_list)):
	# 		temp.append(A_list[i_A]+alpha*B_list[i_A])
	# 	return temp

	def on_train_begin(self, logs={}):
		# if self.isFine:
		# 	self.build_ht_copy_W()
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		
		if epoch in self.startEpoch:
			if epoch == 0:
				ratio = 1
			else:
				ratio = 0.1
			LR = K.get_value(self.model.optimizer.lr)
			K.set_value(self.model.optimizer.lr,LR*ratio)
		
		return

	def on_epoch_end(self, epoch, logs={}):	
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		# if self.isFine:
		# 	for i_m, sub_model in enumerate(self.model.layers):
		# 		if i_m>1:
		# 			if sub_model.name != 'ssr_Cap_model' and sub_model.name != 'ssr_F_Cap_model':
		# 				old_W = self.ht[i_m]
		# 				new_W = sub_model.get_weights()
		# 				delta_W = self.list_operation(new_W,old_W,-1)

		# 				sub_W = self.list_operation(old_W,delta_W,0.1)
		# 				sub_model.set_weights(sub_W)

		# 	self.build_ht_copy_W()
		return
