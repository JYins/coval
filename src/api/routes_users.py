"""User auth routes."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import create_access_token, get_db, hash_password, verify_password
from src.models.user import User


router = APIRouter(prefix="/api/users", tags=["users"])


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: UUID
    email: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


def normalize_email(email: str) -> str:
    value = email.strip().lower()
    if not value:
        raise HTTPException(status_code=400, detail="email is required")
    return value


def validate_password(password: str) -> None:
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="password should be at least 6 chars")


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(data: RegisterRequest, db: Session = Depends(get_db)) -> UserResponse:
    email = normalize_email(data.email)
    validate_password(data.password)

    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="email already registered")

    user = User(email=email, password_hash=hash_password(data.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserResponse(id=user.id, email=user.email)


@router.post("/login", response_model=TokenResponse)
def login_user(data: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    email = normalize_email(data.email)
    user = db.query(User).filter(User.email == email).first()

    if user is None or not verify_password(data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid email or password",
        )

    token = create_access_token({"sub": str(user.id), "email": user.email})
    return TokenResponse(access_token=token, token_type="bearer")

