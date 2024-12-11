import syft as sy


def train(task, xArr, yArr):
    from timm.layers import trunc_normal_
    import torch.nn as nn
    import torch
    import torch.optim as optim
    from tqdm import tqdm
    from sklearn.metrics import roc_curve, auc
    import os
    
    from lib.ViTLike import ViTLike

    
    NUM_EPOCHE = 50
    SEED = 2024
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.makedirs(f'src/result/{SEED}/', exist_ok=True)

    model = ViTLike(embed_dim=2048)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RAdam(
        model.parameters(), lr=0.0002, weight_decay=0.00001)

    for epoch in tqdm(range(NUM_EPOCHE), desc=f'{task}'):
        model.train().to('cuda')
        running_loss = 0.0
        y_arr = []
        Y_arr = []
        for x, y in zip(xArr, yArr):
            optimizer.zero_grad()
            x = x.to('cuda')
            Y = model(x)
            Y_arr.append(Y.detach().cpu()[0])
            y_arr.append(y[0])

            loss = criterion(Y, y.float().to('cuda'))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        fpr, tpr, _ = roc_curve(torch.cat(y_arr), torch.cat(Y_arr))
        auroc = auc(fpr, tpr)

        epcohStr = f'Epoch [{epoch+1}/{NUM_EPOCHE}]'
        lossStr = f'Loss: {running_loss / len(y_arr):.4f}'
        aurocStr = f'auroc: {auroc}'
        str = f'{epcohStr}, {lossStr}, {aurocStr}'
        mode = 'w' if epoch == 0 else 'a'
        with open(f'src/result/{SEED}/{task}.txt', mode, encoding='utf-8') as file:
            file.write(str + '\n')

    # 此处因为模型定义在函数内部，不能序列化，只保留state
    # torch.save(model.state_dict(), f'src/result/{SEED}/{task}.pth')
    torch.save(model, f'src/result/{SEED}/{task}.pth')


for idx in range(4):
    data_site = sy.orchestra.launch(
        name=f"gleason-research-centre-{idx}", reset=False)
    user = data_site.login(email="user@hutb.com", password="syftrocks")
    dataset = user.datasets['Gleason Cancer Biomarker']
    features, targets = dataset.assets

    # train(task=f'user_{idx}', xArr=features.mock, yArr=targets.mock)
    remote_user_code = sy.syft_function_single_use(xArr=features, yArr=targets)(train)
    research_project = user.create_project(
        name="Gleason Cancer Project",
        description='',
        user_email_address="user@hutb.com"
    )
    code_request = research_project.create_code_request(remote_user_code, user)