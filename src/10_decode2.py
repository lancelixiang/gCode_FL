
import torch
import time
from phe import paillier

torch.set_printoptions(precision=4) 

# 生成公钥和私钥
public_key, private_key = paillier.generate_paillier_keypair()
print("公钥:", public_key)
print("私钥:", private_key)

# 假设有一个 2D Tensor
tensor_2d = torch.tensor([ 8.8468e-04,  3.1866e-03,  8.5312e-04, 1.8951e-03, -8.5740e-04,  7.8852e-04,   1.8189e-03,  1.3727e-03,  1.2193e-03,   1.5721e-03,         -
                         3.3975e-04,  8.8055e-04,       1.8635e-03, -2.7619e-04,  1.4676e-03,   2.1140e-03,         -1.2774e-03,  1.3518e-06])
print("原始 2D Tensor:\n", tensor_2d.shape, tensor_2d.dtype)
start_time = time.perf_counter()

# 展平 Tensor 并加密
flatten_values = tensor_2d.tolist()
print('加密前', flatten_values)
encrypted_2d = [public_key.encrypt(x) for x in flatten_values]
end_time = time.perf_counter()
print("加密后:\n", encrypted_2d)
[print(x.ciphertext(), x.exponent) for x in encrypted_2d]
print(f"执行时间: {end_time - start_time} 秒")

# # 解密并恢复形状
decrypted_2d = torch.tensor([private_key.decrypt(x)
                            for x in encrypted_2d]).reshape(tensor_2d.shape)
end_time2 = time.perf_counter()
print("解密后:\n", decrypted_2d.tolist())
print(f"执行时间: {end_time2 - end_time} 秒")
