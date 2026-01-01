"""Audit logging for HIPAA compliance.

Implements comprehensive audit trail as required by HIPAA Security Rule
§164.308(a)(1)(ii)(D) and §164.312(b).
"""

import os
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from sqlalchemy.orm import Session
from ..models.rbac import AuditLog, User

load_dotenv()
MAX_FAILED_ACCESS_ATTEMPTS = int(os.getenv('MAX_FAILED_ACCESS_ATTEMPTS', '100'))

class AuditLogger:
    """Manages audit logging for HIPAA compliance."""

    def __init__(
        self,
        db: Session,
        session_factory: Optional[Callable[[], Session]] = None,
    ) -> None:
        """Initialize audit logger.

        Args:
            db: Database session
            session_factory: Optional session factory for isolated audit transactions
        """
        self.db = db
        self._session_factory = session_factory

    def _persist(self, log: AuditLog) -> None:
        """Persist audit log in an isolated transaction.

        Args:
            log: AuditLog object to persist
        """
        try:
            if self._session_factory:
                with self._session_factory() as s:
                    s.add(log)
                    s.commit()
            else:
                self.db.add(log)
                self.db.commit()
        except Exception:
            pass

    def log_access(
        self,
        user: User,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
        status_code: Optional[int] = None,
        error_message: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        phi_accessed: bool = False,
        phi_fields_accessed: Optional[List[str]] = None,
        justification: Optional[str] = None,
    ) -> AuditLog:
        """Log an access event.

        Args:
            user: User who performed the action
            action: Action performed (e.g., "view", "edit", "delete")
            resource_type: Type of resource accessed (e.g., "patient", "diagnosis")
            resource_id: ID of specific resource
            endpoint: API endpoint accessed
            ip_address: Client IP address
            success: Whether action was successful
            status_code: HTTP status code
            error_message: Error message if action failed
            request_data: Sanitized request parameters
            response_data: Sanitized response summary
            session_id: Session identifier
            phi_accessed: Whether PHI was accessed
            phi_fields_accessed: List of PHI fields accessed
            justification: Business justification for access

        Returns:
            Created AuditLog object
        """
        # Get role names
        role_names = [role.name for role in user.roles]

        # Sanitize request/response data (remove sensitive info)
        safe_request = (
            self._sanitize_data(request_data) if request_data else None
        )
        safe_response = (
            self._sanitize_data(response_data) if response_data else None
        )

        # Create audit log
        # Note: action is intentionally separate from Permission enum.
        log = AuditLog(
            user_id=user.id,
            username=user.username,
            role_names=role_names,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            endpoint=endpoint,
            ip_address=ip_address,
            timestamp=datetime.now(timezone.utc),
            justification=justification,
            success=success,
            status_code=status_code,
            error_message=error_message,
            request_data=safe_request,
            response_data=safe_response,
            session_id=session_id,
            phi_accessed=phi_accessed,
            phi_fields_accessed=phi_fields_accessed or [],
        )

        self._persist(log)
        self.db.refresh(log)

        return log

    def log_login_attempt(
        self,
        username: str,
        success: bool,
        ip_address: Optional[str],
        user_agent: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Log a login attempt.

        Args:
            username: Username that attempted login
            success: Whether login was successful
            ip_address: Client IP address
            user_agent: Client user agent
            error_message: Error message if login failed
        """
        # For failed logins, we may not have a user object
        # Create a simplified audit log entry
        log = AuditLog(
            user_id=None,  # Will be NULL for failed attempts
            username=username,
            role_names=[],
            action='login_attempt',
            resource_type='authentication',
            resource_id=None,
            endpoint='/auth/login',
            ip_address=ip_address,
            timestamp=datetime.now(timezone.utc),
            success=success,
            error_message=error_message,
            request_data={'user_agent': user_agent} if user_agent else None,
            phi_accessed=False,
        )

        self._persist(log)

    def log_logout(
        self,
        user: User,
        session_id: str,
        ip_address: str,
    ) -> None:
        """Log a logout event.

        Args:
            user: User who logged out
            session_id: Session that was terminated
            ip_address: Client IP address
        """
        self.log_access(
            user=user,
            action='logout',
            resource_type='authentication',
            endpoint='/auth/logout',
            ip_address=ip_address,
            session_id=session_id,
            success=True,
            phi_accessed=False,
        )

    def log_permission_denied(
        self,
        user: User,
        action: str,
        resource_type: str,
        resource_id: Optional[str],
        endpoint: str,
        ip_address: str,
        required_permission: str,
    ) -> None:
        """Log a permission denied event.

        Important for detecting potential security breaches.

        Args:
            user: User who was denied access
            action: Action that was attempted
            resource_type: Type of resource
            resource_id: ID of resource
            endpoint: API endpoint
            ip_address: Client IP address
            required_permission: Permission that was required
        """
        self.log_access(
            user=user,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            endpoint=endpoint,
            ip_address=ip_address,
            success=False,
            status_code=403,
            error_message=f'Permission denied: requires {required_permission}',
            phi_accessed=False,
        )

    # TODO: More masking patterns
    def _mask_value(self, value: Any) -> Any:
        """Mask obvious credentials/tokens in strings.

        Args:
            value: Value to potentially mask

        Returns:
            Masked value if it looks like a credential, otherwise original value
        """
        if isinstance(value, str):
            v = value.strip()
            # JWT-like tokens have at least 2 dots
            if v.lower().startswith('bearer ') or v.count('.') >= 2:
                return 'REDACTED'
        return value

    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data for logging.

        Removes or masks sensitive information like passwords, tokens, etc.
        from dictionary data (e.g., request/response objects).

        Note:
            For text string sanitization (PII/secrets redaction), use
            research.auth.scan.detect() which provides advanced PII and
            secret detection using Presidio and regex patterns.

        Args:
            data: Data to sanitize

        Returns:
            Sanitized data
        """
        if not isinstance(data, dict):
            return {}

        sensitive_fields = {
            'password',
            'token',
            'access_token',
            'refresh_token',
            'id_token',
            'jwt',
            'authorization',
            'auth',
            'secret',
            'client_secret',
            'api_key',
            'private_key',
            'session',
            'session_id',
            'cookie',
            'set-cookie',
            'ssn',
            'credit_card',
            'pin',
            'otp',
            'mfa',
        }

        sanitized = {}
        for key, value in data.items():
            # Check if field name suggests sensitive data
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                sanitized[key] = '***REDACTED***'
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_data(item)
                    if isinstance(item, dict)
                    else self._mask_value(item)
                    if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = self._mask_value(value)

        return sanitized

    def get_user_activity(
        self,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditLog]:
        """Get activity logs for a user.

        Args:
            user_id: User ID to get logs for
            start_date: Start date for filtering
            end_date: End date for filtering
            limit: Maximum number of logs to return

        Returns:
            List of audit logs
        """
        query = self.db.query(AuditLog).filter(AuditLog.user_id == user_id)

        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)

        return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()

    def get_resource_access_logs(
        self,
        resource_type: str,
        resource_id: str,
        limit: int = 100,
    ) -> List[AuditLog]:
        """Get access logs for a specific resource.

        Useful for tracking who has accessed a patient's records.

        Args:
            resource_type: Type of resource (e.g., "patient")
            resource_id: ID of resource
            limit: Maximum number of logs to return

        Returns:
            List of audit logs
        """
        return (
            self.db.query(AuditLog)
            .filter(AuditLog.resource_type == resource_type)
            .filter(AuditLog.resource_id == resource_id)
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
            .all()
        )

    def get_failed_access_attempts(
        self,
        start_date: Optional[datetime] = None,
        limit: int = MAX_FAILED_ACCESS_ATTEMPTS,
    ) -> List[AuditLog]:
        """Get failed access attempts.

        Useful for detecting security incidents.

        Args:
            start_date: Start date for filtering
            limit: Maximum number of logs to return

        Returns:
            List of failed audit logs
        """
        query = self.db.query(AuditLog).filter(AuditLog.success.is_(False))

        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)

        return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
