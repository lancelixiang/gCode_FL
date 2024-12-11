import syft as sy


for idx in range(4):
    data_site = sy.orchestra.launch(name=f"gleason-research-centre-{idx}", reset=False)

    # logging in as root client with default credentials
    dataOwner = data_site.login(email="info@openmined.org", password="changethis")

    request = dataOwner.requests[0]
    request.approve()

    data_site.land()