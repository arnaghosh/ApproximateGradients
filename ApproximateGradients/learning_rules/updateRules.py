import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

class UpdateRules():
	def __init__(self,net,dataloader,loss):
		super(UpdateRules,self).__init__()
		self.net = net
		self.dataloader = dataloader
		self.loss = loss

	def update(self,lr,no_update=False):
		if no_update:
			pass
		else:
			for key,param in self.net.named_parameters():
				param -= lr*param.grad

	def computeGradients(self):
		pass
