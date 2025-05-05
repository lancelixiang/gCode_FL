import torch
import time
from phe import paillier

# 生成公钥和私钥
public_key, private_key = paillier.generate_paillier_keypair()
print("公钥:", public_key)
print("私钥:", private_key)

model = torch.load(f'result/88/model_0.pth', weights_only=False)
model0 = torch.load(f'result/88/model_0.pth', weights_only=False)
model1 = torch.load(f'result/88/model_1.pth', weights_only=False)
model2 = torch.load(f'result/88/model_2.pth', weights_only=False)
model3 = torch.load(f'result/88/model_3.pth', weights_only=False)

state_dict = model.state_dict()
state_dict0 = model0.state_dict()
state_dict1 = model1.state_dict()
state_dict2 = model2.state_dict()
state_dict3 = model3.state_dict()

start_time = time.perf_counter()

for k in state_dict:
    state_dict[k] = torch.stack([
        state_dict0[k],
        state_dict1[k],
        state_dict2[k],
        state_dict3[k],
    ]).mean(dim=0)
end_time = time.perf_counter()
print(f"执行时间: {end_time - start_time} 秒")


def en(myTensor):
    # flatten_values = myTensor.flatten().tolist()
    # return torch.tensor([public_key.encrypt(x) for x in flatten_values]).reshape(myTensor.shape)
    flatten_values = myTensor.tolist()
    return torch.tensor([public_key.encrypt(x) for x in flatten_values])

def de(myTensor):
    # flatten_values = myTensor.flatten().tolist()
    # return torch.tensor([private_key.decrypt(x) for x in flatten_values]).reshape(myTensor.shape)
    flatten_values = myTensor.tolist()
    return torch.tensor([private_key.decrypt(x) for x in flatten_values])


# for k in state_dict:
k='slide_head.bias'
print('pppppppppp', state_dict0[k].shape)
state_dict[k] = torch.stack([
    en(state_dict0[k]),
    en(state_dict1[k]),
    en(state_dict2[k]),
    en(state_dict3[k]),
]).mean(dim=0)
end_time_en = time.perf_counter()
print(f"加密执行时间: {end_time_en - end_time} 秒")


# for k in state_dict:
state_dict[k] = de(state_dict[k])
end_time_dn = time.perf_counter()
print(f"解密执行时间: {end_time_dn - end_time_en} 秒")


print('*************************', state_dict[k])
