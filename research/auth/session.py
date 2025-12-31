"""Session management for HIPAA-compliant user authentication.

Implements session handling with automatic timeout and activity tracking
as required by HIPAA Security Rule §164.312(a)(2)(iii).
"""

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from research.models.rbac import User, UserSession

load_dotenv()

class SessionManager:
    """Manages user sessions with HIPAA-compliant security features."""

    # Session timeout: 15 minutes of inactivity (HIPAA recommended)
    # Can be configured via SESSION_TIMEOUT_MINUTES environment variable
    _SESSION_TIMEOUT_MINUTES= int(os.getenv('_SESSION_TIMEOUT_MINUTES', '15'))
    _CLEANUP_EXPIRED_SESSIONS= int(os.getenv('_CLEANUP_EXPIRED_SESSIONS', '30'))
    # Maximum session duration: 8 hours (typical work shift)
    # Can be configured via MAX_SESSION_DURATION_HOURS environment variable
    _MAX_SESSION_DURATION_HOURS= int(os.getenv('_MAX_SESSION_DURATION_HOURS', '8'))

    def __init__(self, db: Session):
        """Initialize session manager.

        Args:
            db: Database session
        """
        self.db = db

    def create_session(
        self,
        user: User,
        ip_address: str,
        user_agent: str,
    ) -> UserSession:
        """Create a new user session.

        Args:
            user: User to create session for
            ip_address: Client IP address
            user_agent: Client user agent string

        Returns:
            New UserSession object
        """
        # Generate secure session token
        session_token = secrets.token_urlsafe(64)

        # Calculate expiration time
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self._MAX_SESSION_DURATION_HOURS)

        # Create session
        session = UserSession(
            user_id=user.id,
            session_token=session_token,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=now,
            expires_at=expires_at,
            last_activity=now,
        )

        self.db.add(session)

        # Update user's last login
        user.last_login = now
        user.failed_login_attempts = 0

        self.db.commit()
        self.db.refresh(session)

        return session

    def increment_failed_login(self, user: User) -> int:
        """Increment failed login attempts counter.

        Should be called when authentication fails. User account will be
        locked after too many failed attempts (typically 5-10).

        Args:
            user: User with failed login attempt

        Returns:
            Updated failed login attempt count
        """
        user.failed_login_attempts += 1
        self.db.commit()
        return user.failed_login_attempts

    def reset_failed_login(self, user: User) -> None:
        """Reset failed login attempts counter.

        Should be called on successful login.

        Args:
            user: User to reset counter for
        """
        user.failed_login_attempts = 0
        self.db.commit()

    def get_session(self, session_token: str) -> Optional[UserSession]:
        """Get active session by token.

        Args:
            session_token: Session token to look up

        Returns:

            UserSession if valid and active, None otherwise
        """
        session = (
            self.db.query(UserSession)
            .filter(UserSession.session_token == session_token)
            .filter(UserSession.revoked.is_(False))
            .first()
        )

        if not session:
            return None

        now = datetime.now(timezone.utc)

        # Check if session has expired
        if session.expires_at < now:
            return None

        # Check for inactivity timeout
        inactivity_limit = now - timedelta(
            minutes=self._SESSION_TIMEOUT_MINUTES
        )
        if session.last_activity and session.last_activity < inactivity_limit:
            return None

        # Update last activity
        session.last_activity = now
        self.db.commit()

        return session

    def revoke_session(self, session_token: str) -> bool:
        """Revoke a session (logout).

        Args:
            session_token: Session token to revoke

        Returns:
            True if session was revoked, False if not found
        """
        session = (
            self.db.query(UserSession)
            .filter(UserSession.session_token == session_token)
            .first()
        )

        if not session:
            return False

        session.revoked = True
        session.revoked_at = datetime.now(timezone.utc)
        self.db.commit()

        return True

    def revoke_all_user_sessions(self, user_id: int) -> int:
        """Revoke all sessions for a user.

        Useful when:
        - Password is changed
        - Account is locked
        - Security incident

        Args:
            user_id: User ID to revoke sessions for

        Returns:
            Number of sessions revoked
        """
        now = datetime.now(timezone.utc)
        count = (
            self.db.query(UserSession)
            .filter(UserSession.user_id == user_id)
            .filter(UserSession.revoked.is_(False))
            .update(
                {'revoked': True, 'revoked_at': now},
                synchronize_session=False,
            )
        )
        self.db.commit()
        return count

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions from database.

        Should be run periodically (e.g., daily) to maintain database hygiene.

        Returns:
            Number of sessions deleted
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self._CLEANUP_EXPIRED_SESSIONS)
        count = (
            self.db.query(UserSession)
            .filter(
                (UserSession.expires_at < now)
                | (UserSession.revoked.is_(True))
            )
            .filter(UserSession.created_at < cutoff)
            .delete(synchronize_session=False)
        )
        self.db.commit()
        return count
