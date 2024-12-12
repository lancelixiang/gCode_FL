import torch

model = torch.load(f'src/result/2024/model_0.pth', weights_only=False)
model0 = torch.load(f'src/result/2024/model_0.pth', weights_only=False)
model1 = torch.load(f'src/result/2024/model_1.pth', weights_only=False)
model2 = torch.load(f'src/result/2024/model_2.pth', weights_only=False)
model3 = torch.load(f'src/result/2024/model_3.pth', weights_only=False)

state_dict = model.state_dict()
state_dict0 = model0.state_dict()
state_dict1 = model1.state_dict()
state_dict2 = model2.state_dict()
state_dict3 = model3.state_dict()

for k in state_dict:
    state_dict[k] = torch.stack([
        state_dict0[k],
        state_dict1[k],
        state_dict2[k],
        state_dict3[k],
    ]).mean(dim=0)
    
torch.save(state_dict, f'src/result/2024/state_dict.pth')