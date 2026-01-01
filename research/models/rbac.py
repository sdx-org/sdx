"""
HIPAA-compliant Role-Based Access Control (RBAC) models.

This module implements database models for user authentication and
authorization following HIPAA Security Rule requirements for access
control and audit trails.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Literal, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.hiperhealth.models.sqla.fhirx import Base


class HealthcareRole(str, Enum):
    """Standard healthcare roles with defined access levels."""

    ADMIN = 'admin'
    DOCTOR = 'doctor'
    NURSE = 'nurse'
    BILLING_CLERK = 'billing_clerk'
    RECEPTIONIST = 'receptionist'
    RESEARCHER = 'researcher'
    AUDITOR = 'auditor'


class Permission(str, Enum):
    """Granular permissions for healthcare data access."""

    # Patient data access
    VIEW_PATIENT_DEMOGRAPHICS = 'view_patient_demographics'
    EDIT_PATIENT_DEMOGRAPHICS = 'edit_patient_demographics'
    VIEW_MEDICAL_RECORDS = 'view_medical_records'
    EDIT_MEDICAL_RECORDS = 'edit_medical_records'
    VIEW_DIAGNOSIS = 'view_diagnosis'
    EDIT_DIAGNOSIS = 'edit_diagnosis'
    VIEW_PRESCRIPTIONS = 'view_prescriptions'
    EDIT_PRESCRIPTIONS = 'edit_prescriptions'
    VIEW_LAB_RESULTS = 'view_lab_results'
    EDIT_LAB_RESULTS = 'edit_lab_results'
    VIEW_MENTAL_HEALTH = 'view_mental_health'
    EDIT_MENTAL_HEALTH = 'edit_mental_health'

    # Financial data access
    VIEW_BILLING = 'view_billing'
    EDIT_BILLING = 'edit_billing'
    PROCESS_PAYMENTS = 'process_payments'

    # Administrative functions
    MANAGE_USERS = 'manage_users'
    MANAGE_ROLES = 'manage_roles'
    VIEW_AUDIT_LOGS = 'view_audit_logs'
    EXPORT_DATA = 'export_data'

    # Scheduling
    VIEW_APPOINTMENTS = 'view_appointments'
    MANAGE_APPOINTMENTS = 'manage_appointments'

    # Research
    VIEW_DEIDENTIFIED_DATA = 'view_deidentified_data'
    EXPORT_RESEARCH_DATA = 'export_research_data'


# Association table for many-to-many relationship between User and Role
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True),
)

# Association table for many-to-many relationship between
# Role and RolePermission
role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True),
    Column(
        'permission_id',
        Integer,
        ForeignKey('permissions.id'),
        primary_key=True,
    ),
)


class User(Base):
    """User model for authentication and RBAC."""

    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Employee/professional identifiers
    employee_id: Mapped[Optional[str]] = mapped_column(
        String(50), unique=True, index=True, nullable=True
    )
    npi_number: Mapped[Optional[str]] = mapped_column(
        String(10), unique=True, index=True, nullable=True
    )  # National Provider Identifier
    license_number: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )
    department: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )

    # Account status
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    is_locked: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    password_changed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    roles: Mapped[List['Role']] = relationship(
        'Role', secondary=user_roles, back_populates='users'
    )
    audit_logs: Mapped[List['AuditLog']] = relationship(
        'AuditLog', back_populates='user'
    )
    sessions: Mapped[List['UserSession']] = relationship(
        'UserSession', back_populates='user'
    )

class Role(Base):
    """Role model for RBAC."""

    __tablename__ = 'roles'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Track if this is a system-defined role
    is_system: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    users: Mapped[List['User']] = relationship(
        'User', secondary=user_roles, back_populates='roles'
    )
    permissions: Mapped[List['RolePermission']] = relationship(
        'RolePermission', secondary=role_permissions, back_populates='roles'
    )


class RolePermission(Base):
    """Permission model for fine-grained access control."""

    __tablename__ = 'permissions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resource: Mapped[
        Optional[
            Literal[
                'patient',
                'billing',
                'user',
                'appointment',
                'research',
                'audit',
            ]
        ]
    ] = mapped_column(String(100), nullable=True)
    action: Mapped[
        Optional[Literal['view', 'edit', 'delete', 'manage', 'export']]
    ] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    roles: Mapped[List['Role']] = relationship(
        'Role', secondary=role_permissions, back_populates='permissions'
    )


class UserSession(Base):
    """Track user sessions for audit and security purposes."""

    __tablename__ = 'user_sessions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey('users.id'), nullable=False
    )
    session_token: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )

    # Session metadata
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45), nullable=True
    )  # IPv6 compatible
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Session lifecycle
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_activity: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )

    # Relationships
    user: Mapped['User'] = relationship('User', back_populates='sessions')


class AuditLog(Base):
    """Audit log for tracking all access to PHI.

    Required by HIPAA Security Rule §164.308(a)(1)(ii)(D) and
    §164.312(b).
    """

    __tablename__ = 'audit_logs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Who
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey('users.id'), nullable=False
    )
    username: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # Denormalized for performance
    role_names: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON string of role names at time of access

    # What
    action: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )
    resource_type: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )
    resource_id: Mapped[Optional[str]] = mapped_column(
        String(255), index=True, nullable=True
    )

    # Where
    endpoint: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45), nullable=True
    )

    # When
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    # Why (optional - business justification)
    justification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Outcome
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Additional context
    request_data: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON string of sanitized request parameters
    response_data: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON string of sanitized response summary
    session_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    # Data accessed (for minimum necessary compliance)
    phi_accessed: Mapped[bool] = mapped_column(Boolean, default=False)
    phi_fields_accessed: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON string of PHI fields accessed

    # Relationships
    user: Mapped['User'] = relationship('User', back_populates='audit_logs')
