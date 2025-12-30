"""Authentication and authorization package for HIPAA-compliant access control."""

from research.auth.dependencies import (
        get_current_active_user,
        get_current_user,
        require_permission,
        require_role
)

from research.auth.password import get_password_hash, verify_password
from research.auth.rbac import RBACManager
from research.auth.session import SessionManager

__all__ = [
    'get_current_user',
    'get_current_active_user',
    'require_permission',
    'require_role',
    'get_password_hash',
    'verify_password',
    'RBACManager',
    'SessionManager',
]
