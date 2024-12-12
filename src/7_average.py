import torch
import pandas as pd
from sklearn.metrics import roc_curve, auc
import torch.nn as nn

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

# 测试性能开始
model.load_state_dict(state_dict)
SLIDE_DATA = pd.read_csv(f'src/data.csv', index_col=0)
data = SLIDE_DATA.loc[:, 'test'].dropna()
featuresArr = []
labelArr = []
for id in data:
    insE = torch.load(
        f'dataset/efficientnet_b0/{id}.pth', weights_only=False)
    insM = torch.load(
        f'dataset/MambaVision-S-1K/{id}.pth', weights_only=False)
    features1 = torch.transpose(insE['features'], 1, 0)
    features2 = torch.transpose(insM['features'], 1, 0)
    features = torch.cat((features1, features2), dim=2)
    featuresArr.append(features)
    labelArr.append(torch.tensor([insE['label']]))
    
criterion = nn.BCEWithLogitsLoss()
model.eval().to('cuda')
running_loss = 0.0
y_arr = []
Y_arr = []
for x, y in zip(featuresArr, labelArr):
    x = x.to('cuda')
    Y = model(x)

    Y_arr.append(Y.detach().cpu()[0])
    y_arr.append(y[0])

    loss = criterion(Y, y.float().to('cuda'))
    running_loss += loss.item()

fpr, tpr, _ = roc_curve(torch.cat(y_arr), torch.cat(Y_arr))
auroc = auc(fpr, tpr)

lossStr = f'test Loss: {running_loss / len(y_arr):.4f}'
aurocStr = f'test auroc: {auroc}'
str = f'{lossStr}, {aurocStr}'
print(str)

SEED = 2024
with open(f'src/result/{SEED}/flavg.txt', 'a', encoding='utf-8') as file:
    file.write(str + '\n')