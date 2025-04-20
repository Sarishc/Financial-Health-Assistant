# app/api/auth/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Create router
router = APIRouter()

# Setup password encryption
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Setup OAuth2 with Bearer token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/token")

# Simulated user database (in a real app, this would be a database)
fake_users_db = {
    "john@example.com": {
        "email": "john@example.com",
        "full_name": "John Doe",
        "hashed_password": pwd_context.hash("secret123"),
        "disabled": False,
    },
    "alice@example.com": {
        "email": "alice@example.com",
        "full_name": "Alice Smith",
        "hashed_password": pwd_context.hash("password456"),
        "disabled": False,
    }
}

# Simple user model
class User:
    def __init__(self, email, full_name=None, disabled=None):
        self.email = email
        self.full_name = full_name
        self.disabled = disabled

def verify_password(plain_password, hashed_password):
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Get password hash"""
    return pwd_context.hash(password)

def get_user(db, email: str):
    """Get user from database"""
    if email in db:
        user_dict = db[email]
        return User(
            email=user_dict["email"],
            full_name=user_dict["full_name"],
            disabled=user_dict["disabled"]
        )
    return None

def authenticate_user(db, email: str, password: str):
    """Authenticate a user"""
    user = get_user(db, email)
    if not user:
        return False
    if not verify_password(password, db[email]["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, "your-secret-key", algorithm="HS256")
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(fake_users_db, email=email)
    if user is None:
        raise credentials_exception
    
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Check if the user is active"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user