from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
import jwt
from dotenv import load_dotenv
import os
import base64

load_dotenv()

app = FastAPI()

SECRET_KEY = base64.b64decode(os.getenv("ISSUER_SECRET"))
ALGORITHM = "HS512"

print(f"=== FastAPI Starting ===")
print(f"SECRET_KEY loaded: {'YES' if SECRET_KEY else 'NO'}")
if SECRET_KEY:
    print(f"SECRET_KEY first 10 chars: {SECRET_KEY[:10]}...")


class RecommendationRequest(BaseModel):
    nurse_id: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"\n=== Incoming Request ===")
    print(f"Method: {request.method}")
    print(f"URL: {request.url}")
    print(f"Headers: {dict(request.headers)}")
    response = await call_next(request)
    print(f"Response status: {response.status_code}")
    return response


@app.post("/recommend")
def recommend(req: RecommendationRequest, authorization: str = Header(None)):
    print(f"\n=== /recommend endpoint called ===")
    print(f"Authorization header: {authorization}")
    
    if not authorization:
        print("ERROR: No authorization header!")
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if not authorization.startswith("Bearer "):
        print(f"ERROR: Invalid format. Got: {authorization[:50]}")
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    token = authorization.split(" ")[1]
    print(f"Token (first 30 chars): {token[:30]}...")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(f"✓ Token decoded successfully!")
        print(f"Payload: {payload}")
    except jwt.ExpiredSignatureError:
        print("ERROR: Token expired!")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        print(f"ERROR: Invalid token - {e}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    
    return {
        "shift_ids": ["dummy-1", "dummy-2", "dummy-3"],
        "received_nurse": req.nurse_id
    }