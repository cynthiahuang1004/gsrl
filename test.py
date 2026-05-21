import torch
state = torch.load('/media/hdd/ihsuan/gsrl/datasets/checkpoints/SITR_B18.pth', map_location='cpu')
print(type(state))
print(list(state.keys())[:10])