import base64
import os
import jwt
from dotenv import load_dotenv
from datetime import datetime, timedelta
import uuid

load_dotenv()

# Läs secret key
SECRET_KEY = base64.b64decode(os.getenv("ISSUER_SECRET", "dG9wU2VjcmV0MTIz"))
ALGORITHM = "HS512"

def generate_token(sub: str = None, expire_minutes: int = 60):
    """Genererar ett JWT för test/development"""
    if sub is None:
        sub = str(uuid.uuid4())  

    now = datetime.utcnow()
    payload = {
        "sub": sub,
        "iat": now,
        "exp": now + timedelta(minutes=expire_minutes)
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

if __name__ == "__main__":
    token = generate_token()
    print("=== Generated token ===")
    print(token)