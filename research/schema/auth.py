"""Pydantic schemas for authentication and user management."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field

from research.models.rbac import HealthcareRole, Permission


class UserLogin(BaseModel):
    """Schema for user login."""

    username: str
    password: str

class UserCreate(UserLogin):
    """Schema for creating a new user.

    Inherits username and password from UserLogin to avoid duplication.
    """

    username: str = Field(..., min_length=3, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=12)
    full_name: str = Field(..., min_length=1, max_length=255)
    employee_id: Optional[str] = None
    npi_number: Optional[str] = Field(None, min_length=10, max_length=10)
    license_number: Optional[str] = None
    department: Optional[str] = None
    role_names: List[str] = Field(default_factory=list)


class UserUpdate(BaseModel):
    """Schema for updating user information."""

    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    department: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """Schema for user response."""

    id: int
    username: str
    email: str
    full_name: str
    employee_id: Optional[str]
    npi_number: Optional[str]
    department: Optional[str]
    is_active: bool
    is_locked: bool
    created_at: datetime
    last_login: Optional[datetime]
    roles: List[str]

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class PasswordChange(BaseModel):
    """Schema for password change."""

    current_password: str
    new_password: str = Field(..., min_length=12)

class RoleAssignment(BaseModel):
    """Schema for assigning/removing roles."""

    user_id: int
    role_name: HealthcareRole = Field(
        ..., description='Healthcare role from HealthcareRole enum'
    )


class PermissionCheck(BaseModel):
    """Schema for checking permissions.

    Uses Permission enum directly for type safety and validation.
    Accepts both enum members and their string values.
    """

    user_id: int
    permission: Permission = Field(
        ..., description='Permission from Permission enum'
    )


class AuditLogResponse(BaseModel):
    """Schema for audit log response."""

    id: int
    username: str
    role_names: List[str]
    action: str
    resource_type: str
    resource_id: Optional[str]
    timestamp: datetime
    success: bool
    ip_address: Optional[str]
    phi_accessed: bool

    class Config:
        """Pydantic configuration."""

        from_attributes = True
