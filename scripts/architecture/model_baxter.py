import torch
import torch.nn as nn
from torch.autograd import Variable

# DMLP Model-Path Generator
class MLP(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLP, self).__init__()
		self.fc = nn.Sequential(
                    nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(),
                    nn.Linear(1280, 896), nn.PReLU(), nn.Dropout(),
                    nn.Linear(896, 512), nn.PReLU(), nn.Dropout(),
                    nn.Linear(512, 384), nn.PReLU(), nn.Dropout(),
                    nn.Linear(384, 256), nn.PReLU(), nn.Dropout(),
                    nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
                    nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
                    nn.Linear(64, 32), nn.PReLU(),
                    nn.Linear(32, output_size))

	def forward(self, x):
		out = self.fc(x)
		return out
