from datetime import datetime, timedelta
from typing import Union, Any
import jwt
from utils.util import load_config, predict

CONFIG_PATH = "configs/config.yaml"


appsettings = load_config(CONFIG_PATH)
SECURITY_ALGORITHM = appsettings["SECURITY_ALGORITHM"] 
SECRET_KEY =  appsettings["SECRET_KEY"] 
def generate_token(username: Union[str, Any]) -> str:
    expire = datetime.utcnow() + timedelta(
        seconds=60 * 60 * 24 * 3  # Expired after 3 days
    )
    to_encode = {
        "exp": expire, "username": username
    }
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=SECURITY_ALGORITHM)
    return encoded_jwt