import yaml
from yaml.loader import SafeLoader
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Pre-hashing all plain text passwords once
# stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

try:
    authenticator.login(max_concurrent_users=2)
except Exception as e:
    st.error(e)

def hash(password:str):
    hashed_pw = stauth.Hasher.hash(password)
    print(hashed_pw)
    return hashed_pw
