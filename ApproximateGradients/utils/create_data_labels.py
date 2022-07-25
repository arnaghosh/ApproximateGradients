import os
import numpy as np
import torch 
import torch.nn as nn
from tqdm import tqdm

from .models import LinearNN, MultiLayerNN

def create_linear_data_labels(inp_dim=10,out_dim=1,num_data=100,seed=13):
	"""
	Creates  data and label tensors using a linear single layerNN mapping
	Args:
		inp_dim: Input dimensionality
		out_dim: Output dimensionality
		num_data: Number of data points
		seed: seed for reproducibility
	Returns:
		inp_tensor: Input tensor of shape (num_data,inp_dim)
		out_tensor: Output tensor of shape (num_data,out_dim)
		teacher_net: nn.Module object that defines the mapping between inp_tensor and out_tensor
	"""
	torch.manual_seed(seed)
	np.random.seed(seed)
	inp_tensor = torch.rand((num_data,inp_dim))
	inp_tensor.requires_grad_ = False
	teacher_net = LinearNN(inp_size=inp_dim,out_size=out_dim)
	with torch.no_grad():
		out_tensor = teacher_net(inp_tensor)
	return inp_tensor,out_tensor,teacher_net

def create_NL_data_labels(inp_dim=10,out_dim=1,num_data=100,num_layers_teacher=1,hidden_dim_teacher=None,non_linearity_teacher=None,seed=13):
	"""
	Creates  data and label tensors using a nonlinear multi-layer NN mapping
	Args:
		inp_dim: Input dimensionality
		out_dim: Output dimensionality
		num_data: Number of data points
		num_layers_teacher: Number of layers in teacher network
		hidden_dim_teacher: Hidden layer dimensionality in teacher network
		non_linearity_teacher: Non-linearity for teacher network
		seed: seed for reproducibility
	Returns:
		inp_tensor: Input tensor of shape (num_data,inp_dim)
		out_tensor: Output tensor of shape (num_data,out_dim)
		teacher_net: nn.Module object that defines the mapping between inp_tensor and out_tensor
	"""
	torch.manual_seed(seed)
	np.random.seed(seed)
	inp_tensor = torch.rand((num_data,inp_dim))
	inp_tensor.requires_grad_ = False
	teacher_net = MultiLayerNN(inp_size=inp_dim,
								num_layers=num_layers_teacher,
								hidden_dim=hidden_dim_teacher,
								non_linearity=non_linearity_teacher,
								out_size=out_dim)
	with torch.no_grad():
		out_tensor = teacher_net(X)
	return inp_tensor,out_tensor,teacher_net