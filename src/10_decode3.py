import torch
import syft as sy

# 初始化加密环境
# 不再需要 sy.TorchHook(torch)
# alice = sy.VirtualWorker(id="alice")  # 直接创建虚拟Worker

# 生成密钥对（Paillier）
public_key, private_key = sy.frameworks.torch.he.paillier.keygen()

# 加密张量
x = torch.tensor([1.0, 2.0, 3.0])
x_enc = x.encrypt(public_key)  # 返回 PaillierTensor

# 保存加密张量
# torch.save(x_enc, "encrypted_tensor.pt")

# # 加载加密张量（需保留私钥！）
# loaded_enc = torch.load("encrypted_tensor.pt")