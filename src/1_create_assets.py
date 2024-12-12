import syft as sy
import pandas as pd
import numpy as np
import torch

SEED = 2024
np.random.seed(SEED)

SLIDE_DATA = pd.read_csv(f'src/data.csv', index_col=0)


for idx in range(4):
    data_site = sy.orchestra.launch(
        name=f"gleason-research-centre-{idx}", reset=True)
    dataOwner = data_site.login(
        email="info@openmined.org", password="changethis")

    data = SLIDE_DATA.loc[:, f'client{idx}'].dropna()
    featuresArr = []
    featuresArr_mock = []
    labelArr = []
    labelArr_mock = []
    for id in data:
        insE = torch.load(
            f'dataset/efficientnet_b0/{id}.pth', weights_only=False)
        insM = torch.load(
            f'dataset/MambaVision-S-1K/{id}.pth', weights_only=False)
        features1 = torch.transpose(insE['features'], 1, 0)
        features2 = torch.transpose(insM['features'], 1, 0)
        features = torch.cat((features1, features2), dim=2)
        featuresArr.append(features)
        featuresArr_mock.append(torch.from_numpy(
            np.random.uniform(low=-1, size=features.shape)).to(torch.float))
        labelArr.append(torch.tensor([insE['label']]))
        labelArr_mock.append(torch.tensor([[0, 1, 1, 0]]))

    features_asset = sy.Asset(
        name="Gleason Cancer Data: Features",
        data=featuresArr,      # real data
        mock=featuresArr_mock[0:3]  # mock data
    )
    targets_asset = sy.Asset(
        name="Gleason Cancer Data: Targets",
        data=labelArr,      # real data
        mock=labelArr_mock[0:3]  # mock data
    )

    gleason_cancer_dataset = sy.Dataset(
        name="Gleason Cancer Biomarker",
        description='',
        summary='',
        citation='',
        url='',
    )
    gleason_cancer_dataset.add_asset(features_asset)
    gleason_cancer_dataset.add_asset(targets_asset)

    dataOwner.upload_dataset(dataset=gleason_cancer_dataset)

    data_site.land()
