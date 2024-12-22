import syft as sy


def train(xArr, yArr):
    import torch.nn as nn
    import torch
    import torch.optim as optim
    from tqdm import tqdm
    from sklearn.metrics import roc_curve, auc
    import os

    from lib.ViTLike import ViTLike

    NUM_EPOCHE = 10
    SEED = 88
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.makedirs(f'src/result/{SEED}/', exist_ok=True)
    STAGE = 'MOCK' if os.path.exists(f'src/result/{SEED}/auc.txt') else 'REAL'

    model = ViTLike(embed_dim=2048)
    if os.path.exists(f'src/result/{SEED}/state_dict.pth'):
        state_dict = torch.load(f'src/result/{SEED}/state_dict.pth', weights_only=True)
        model.load_state_dict(state_dict)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RAdam(
        model.parameters(), lr=0.0002, weight_decay=0.00001)

    for epoch in tqdm(range(NUM_EPOCHE)):
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

        if STAGE == 'REAL':
            mode = 'w' if epoch == 0 else 'a'
            with open(f'src/result/{SEED}/auc.txt', mode, encoding='utf-8') as file:
                file.write(str + '\n')

    if STAGE == 'REAL':
        torch.save(model, f'src/result/{SEED}/model.pth')


for idx in range(4):
    data_site = sy.orchestra.launch(
        name=f"gleason-research-centre-{idx}", reset=False)
    user = data_site.login(email="user@hutb.com", password="syftrocks")
    dataset = user.datasets['Gleason Cancer Biomarker']
    features, targets = dataset.assets

    # import os
    # SEED = 88
    # train(xArr=features.mock, yArr=targets.mock)
    # os.rename(f'src/result/{SEED}/auc.txt', f'src/result/{SEED}/auc_{idx}.txt')
    # os.rename(f'src/result/{SEED}/model.pth', f'src/result/{SEED}/model_{idx}.pth')
    
    remote_user_code = sy.syft_function_single_use(
        xArr=features, yArr=targets)(train)
    research_project = sy.Project(
        name="Gleason Cancer Project",
        description='',
        members=[user]
    )

    code_request = research_project.create_code_request(remote_user_code, user)
    data_site.land()
