import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

from .updateRules import UpdateRules

class GD(UpdateRules):
	def __init__(self,net,dataloader,loss):
		super(GD,self).__init__(net,dataloader,loss)

	def computeGradients(self):
		# Overloading superclass function
		self.net.zero_grad() 	# zero out gradients at the start and accumulate gradients over dataloader
		for x,y in self.dataloader:
			out = self.net(x)
			loss_val = self.loss(out,y)
			loss_val.backward()
		grad_list = []
		for key,param in self.net.named_parameters():
			grad_list.append(param.grad.data.flatten())
		grad_vector = torch.cat(grad_list)
		return grad_vector


class SGD(UpdateRules):
	def __init__(self,net,dataloader,loss):
		super(GD,self).__init__(net,dataloader,loss)

	def computeGradients(self):
		# Overloading superclass function
		grad_vector_list = []
		for x,y in self.dataloader:
			self.net.zero_grad()
			out = self.net(x)
			loss_val = self.loss(out,y)
			loss_val.backward()
			grad_list = []
			for key,param in self.net.named_parameters():
				grad_list.append(param.grad.data.flatten())
			grad_vector = torch.cat(grad_list)
			grad_vector_list.append(grad_vector)
		return grad_vector

