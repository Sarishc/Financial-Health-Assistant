# app/api/schemas/messages.py
from pydantic import BaseModel

class Message(BaseModel):
    """Basic message response schema"""
    message: str