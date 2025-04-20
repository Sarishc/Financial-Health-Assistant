# app/api/routes/users.py
from fastapi import APIRouter, Depends, HTTPException, Body, Path
from typing import List, Optional
from app.api.auth.auth import get_current_active_user, get_password_hash
from app.api.schemas.auth import User, UserInDB
from app.api.schemas.messages import Message

# Simulated user database (in a real app, this would be a database)
from app.api.auth.auth import fake_users_db

# Create router
router = APIRouter()

@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information
    """
    return current_user

@router.put("/users/me/update", response_model=User)
async def update_user(
    full_name: Optional[str] = Body(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update current user information
    
    - **full_name**: New full name for the user
    """
    # Get user from database
    if current_user.email not in fake_users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update fields
    if full_name is not None:
        fake_users_db[current_user.email]["full_name"] = full_name
    
    # Return updated user
    return User(
        email=current_user.email,
        full_name=fake_users_db[current_user.email]["full_name"],
        disabled=fake_users_db[current_user.email]["disabled"]
    )

@router.put("/users/me/change-password", response_model=Message)
async def change_password(
    current_password: str = Body(...),
    new_password: str = Body(..., min_length=8),
    current_user: User = Depends(get_current_active_user)
):
    """
    Change user password
    
    - **current_password**: Current password
    - **new_password**: New password (minimum 8 characters)
    """
    from app.api.auth.auth import verify_password
    
    # Get user from database
    if current_user.email not in fake_users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Verify current password
    if not verify_password(current_password, fake_users_db[current_user.email]["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect current password")
    
    # Update password
    fake_users_db[current_user.email]["hashed_password"] = get_password_hash(new_password)
    
    return {"message": "Password updated successfully"}

@router.put("/users/me/disable", response_model=Message)
async def disable_account(
    current_user: User = Depends(get_current_active_user)
):
    """
    Disable user account
    """
    # Get user from database
    if current_user.email not in fake_users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Disable account
    fake_users_db[current_user.email]["disabled"] = True
    
    return {"message": "Account disabled successfully"}

@router.get("/users/preferences", response_model=dict)
async def get_user_preferences(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get user preferences
    """
    # In a real application, this would fetch preferences from a database
    # Here we return some sample preferences
    
    return {
        "currency": "USD",
        "dateFormat": "MM/DD/YYYY",
        "theme": "light",
        "notifications": {
            "email": True,
            "push": False
        },
        "dashboardWidgets": [
            "spending_summary",
            "category_breakdown",
            "top_recommendations"
        ]
    }

@router.put("/users/preferences", response_model=dict)
async def update_user_preferences(
    preferences: dict = Body(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update user preferences
    
    - **preferences**: Dictionary of user preferences
    """
    # In a real application, this would update preferences in a database
    # Here we just return the provided preferences
    
    return preferences