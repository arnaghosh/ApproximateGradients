import os
import numpy as np
import torch 
import torch.nn as nn
from tqdm import tqdm

from .models import LinearNN, MultiLayerNN

def load_student_net(inp_dim,out_dim=1,seed=13):
	"""
	Creates a linear single layer student network
	Args:
		inp_dim: Input dimensionality
		out_dim: Output dimensionality
		seed: random seed to ensure reproducibility
	"""
	torch.manual_seed(seed)
	np.random.seed(seed)
	student_net = LinearNN(inp_size=inp_dim,out_size=out_dim)
	return student_net

def load_multilayer_student_net(inp_dim,num_layers=1,hidden_dim=None,non_linearity=None,out_dim=1,seed=13):
	"""
	Creates a nonlinear multi layer student network
	Args:
		inp_dim: Input dimensionality
		num_layers: Number of layers in model
		hidden_dim: Hidden layer dimensionality
		non_linearity: Non-linearity to be applied
		out_dim: Output dimensionality
		seed: random seed to ensure reproducibility
	"""
	torch.manual_seed(seed)
	np.random.seed(seed)
	student_net = MultiLayerNN(inp_size=inp_dim,out_size=out_dim,num_layers=num_layers,hidden_dim=hidden_dim,non_linearity=non_linearity)
	return student_net

def copy_from_model_to_model(net1,net2):
	"""
	Copies all params from one network to another
	Args:
		net1: Network (nn.Module object) to copy param values from
		net2: Network (nn.Module object) to copy param values to
	"""
	assert type(net1).__name__==type(net2).__name__, "Cannot copy params from {} to {}".format(type(net1).__name__,type(net2).__name__)
	reference_params_list = list(net1.parameters())
	new_params_list = list(net2.parameters())
	assert len(reference_params_list)==len(new_params_list), "Param lists ({} vs {}) don't match!".format(len(reference_params_list),len(new_params_list))
	for pidx,p in enumerate(new_params_list):
		p.data = reference_params_list[pidx].data.clone()

def check_model_params_equal(net1,net2):
	"""
	Checks if all params of one network are equal to another
	Args:
		net1, net2: Networks (nn.Module objects) to compare
	"""
	if type(net1).__name__!=type(net2).__name__:
		return False
	reference_params_list = list(net1.parameters())
	check_params_list = list(net2.parameters())
	if len(reference_params_list)!= len(check_params_list):
		return False
	for pidx,p in enumerate(check_params_list):
		if p.data.shape!=reference_params_list[pidx].data.shape:
			return False
		if not torch.allclose(p.data,reference_params_list[pidx].data):
			return False
	return True