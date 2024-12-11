import syft as sy
import os

SEED = 2024

for idx in range(4):
    data_site = sy.orchestra.launch(
        name=f"gleason-research-centre-{idx}", reset=False)
    user = data_site.login(email="user@hutb.com", password="syftrocks")
    dataset = user.datasets['Gleason Cancer Biomarker']
    features, targets = dataset.assets

    user.code.train(xArr=features, yArr=targets).get()

    os.rename(f'src/result/{SEED}/auc.txt', f'src/result/{SEED}/auc_{idx}.txt')
    os.rename(f'src/result/{SEED}/model.txt', f'src/result/{SEED}/model_{idx}.txt')

    data_site.land()
