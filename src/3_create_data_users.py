import syft as sy


for idx in range(4):
    data_site = sy.orchestra.launch(name=f"gleason-research-centre-{idx}", reset=False)

    # logging in as root client with default credentials
    dataOwner = data_site.login(email="info@openmined.org", password="changethis")

    user_account_info = dataOwner.users.create(
        email="user@hutb.com",
        name="Dr. Lance Lee",
        password="syftrocks",
        password_verify="syftrocks",
        institution="Data Science Institute",
        website="https://datascience_institute.research.data"
    )
    
    print(f"New User: {user_account_info.name} ({user_account_info.email}) registered as {user_account_info.role}")
    data_site.land()