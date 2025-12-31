"""FastAPI dependencies for authentication and authorization.

Provides dependency injection functions for protecting endpoints
with RBAC and audit logging.
"""

from functools import wraps
from typing import Callable,Optional

from fastapi import Cookie, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from research.app.database import SessionLocal
from research.auth.audit import AuditLogger
from research.auth.rbac import RBACManager
from research.auth.session import SessionManager
from research.models.rbac import Permission, User


def get_db():
    """Get database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    request: Request,
    session_token: Optional[str] = Cookie(None, alias='session_token'),
    db: Session = Depends(get_db),
) -> User:
    """Get current authenticated user from session token.

    Args:
        request: FastAPI request object
        session_token: Session token from cookie
        db: Database session

    Returns:
        Authenticated User object

    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Could not validate credentials',
        headers={'WWW-Authenticate': 'Bearer'},
    )

    if not session_token:
        raise credentials_exception

    # Get session
    session_manager = SessionManager(db)
    session = session_manager.get_session(session_token)

    if not session:
        raise credentials_exception

    # Get user
    user = db.query(User).filter(User.id == session.user_id).first()

    if not user:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user.

    Args:
        current_user: Current authenticated user

    Returns:
        User object if active

    Raises:
        HTTPException: If user is inactive or locked
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Inactive user account',
        )

    if current_user.is_locked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Account is locked. Contact administrator.',
        )

    return current_user


def require_permission(permission: Permission):
    """Dependency factory to require a specific permission.

    Usage:
        @app.get("/patients/{patient_id}")
        async def get_patient(
            patient_id: str,
            user: User = Depends(require_permission(Permission.VIEW_PATIENT_DEMOGRAPHICS))
        ):
            ...

    Args:
        permission: Required permission

    Returns:
        Dependency function that checks permission
    """

    async def permission_checker(
        request: Request,
        user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db),
    ) -> User:
        """Check if user has required permission."""
        rbac = RBACManager(db)

        if not rbac.user_has_permission(user, permission):
            # Log permission denied
            audit = AuditLogger(db)
            audit.log_permission_denied(
                user=user,
                action=permission.value.split('_')[0],  # e.g., "view"
                resource_type=permission.value.split('_', 1)[1]
                if '_' in permission.value
                else '',
                resource_id=None,
                endpoint=str(request.url.path),
                ip_address=request.client.host if request.client else "unknown",
                required_permission=permission.value,
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f'Permission denied: requires {permission.value}',
            )

        return user

    return permission_checker


def require_role(role_name: str):
    """Dependency factory to require a specific role.

    Usage:
        @app.get("/admin/users")
        async def list_users(
            user: User = Depends(require_role("admin"))
        ):
            ...

    Args:
        role_name: Required role name

    Returns:
        Dependency function that checks role
    """

    async def role_checker(
        request: Request,
        user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db),
    ) -> User:
        """Check if user has required role."""
        rbac = RBACManager(db)

        if not rbac.user_has_role(user, role_name):
            # Log permission denied
            audit = AuditLogger(db)
            audit.log_permission_denied(
                user=user,
                action='access',
                resource_type='role_restricted',
                resource_id=None,
                endpoint=str(request.url.path),
                ip_address=request.client.host if request.client else "unknown",
                required_permission=f'role:{role_name}',
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: requires role '{role_name}'",
            )

        return user

    return role_checker


def require_any_permission(*permissions: Permission):
    """Dependency factory to require any of the specified permissions.

    Args:
        permissions: One or more permissions (user needs at least one)

    Returns:
        Dependency function that checks permissions
    """

    async def permission_checker(
        request: Request,
        user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db),
    ) -> User:
        """Check if user has any of the required permissions."""
        rbac = RBACManager(db)

        has_permission = any(
            rbac.user_has_permission(user, perm) for perm in permissions
        )

        if not has_permission:
            # Log permission denied
            audit = AuditLogger(db)
            perm_names = [p.value for p in permissions]
            audit.log_permission_denied(
                user=user,
                action='access',
                resource_type='multi_permission',
                resource_id=None,
                endpoint=str(request.url.path),
                ip_address=request.client.host if request.client else "unknown",
                required_permission=f'any_of:{",".join(perm_names)}',
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f'Permission denied: requires one of {perm_names}',
            )

        return user

    return permission_checker


def audit_access(
    action: str,
    resource_type: str,
    phi_accessed: bool = False,
):
    """Decorator to automatically audit endpoint access.

    Usage:
        @app.get("/patients/{patient_id}")
        @audit_access("view", "patient", phi_accessed=True)
        async def get_patient(patient_id: str, user: User = Depends(get_current_user)):
            ...

    Args:
        action: Action being performed
        resource_type: Type of resource being accessed
        phi_accessed: Whether PHI is being accessed

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract dependencies from kwargs
            request = kwargs.get('request')
            user = kwargs.get('user')
            db = kwargs.get('db')

            # Extract resource_id from path parameters if available
            resource_id = kwargs.get(f'{resource_type}_id')

            if user and db and request:
                audit = AuditLogger(db)

                try:
                    # Execute endpoint
                    result = await func(*args, **kwargs)

                    # Log successful access
                    audit.log_access(
                        user=user,
                        action=action,
                        resource_type=resource_type,
                        resource_id=str(resource_id) if resource_id else None,
                        endpoint=str(request.url.path),
                        ip_address=request.client.host
                        if request.client
                        else None,
                        success=True,
                        status_code=200,
                        phi_accessed=phi_accessed,
                    )

                    return result

                except Exception as e:
                    # Log failed access
                    audit.log_access(
                        user=user,
                        action=action,
                        resource_type=resource_type,
                        resource_id=str(resource_id) if resource_id else None,
                        endpoint=str(request.url.path),
                        ip_address=request.client.host
                        if request.client
                        else None,
                        success=False,
                        error_message=str(e),
                        phi_accessed=phi_accessed,
                    )
                    raise
            else:
                # If dependencies not available, just execute
                return await func(*args, **kwargs)

        return wrapper

    return decorator
