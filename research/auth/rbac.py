"""Role-Based Access Control (RBAC) manager.

Implements HIPAA-compliant access control logic for managing
permissions and enforcing the minimum necessary rule.
"""

from typing import List, Set
from sqlalchemy.orm import Session
from research.models._valid_mappings import ROLE_PERMISSION_DEFAULTS
from research.models.rbac import (
    HealthcareRole,
    Permission,
    Role,
    RolePermission,
    User,
)


class RBACManager:
    """Manages role-based access control operations."""

    def __init__(self, db: Session):
        """Initialize RBAC manager.

        Args:
            db: Database session
        """
        self.db = db

    @staticmethod
    def _activity_condition(user: User) -> bool:
        return not user.is_active or user.is_locked

    def user_has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has a specific permission.

        Implements the minimum necessary rule by checking if user's roles
        grant the required permission.

        Args:
            user: User to check permissions for
            permission: Permission to check

        Returns:
            True if user has permission, False otherwise
        """
        if self._activity_condition(user):
            return False

        # Get all permissions for user's roles
        user_permissions: set[str] = self.get_user_permissions(user)
        return permission.value in user_permissions

    def user_has_role(self, user: User, role_name: str) -> bool:
        """Check if user has a specific role.

        Args:
            user: User to check roles for
            role_name: Role name to check

        Returns:
            True if user has role, False otherwise
        """
        if self._activity_condition(user):
            return False

        return any(role.name == role_name for role in user.roles)

    def get_user_permissions(self, user: User) -> Set[str]:
        """Get all permissions for a user.

        Aggregates permissions from all of the user's roles.

        Args:
            user: User to get permissions for

        Returns:
            Set of permission names
        """
        return {
            permission.name
            for role in user.roles
            for permission in role.permissions
        }

    def get_user_roles(self, user: User) -> List[str]:
        """Get all role names for a user.

        Args:
            user: User to get roles for

        Returns:
            List of role names
        """
        return [role.name for role in user.roles]

    def assign_role(self, user: User, role_name: str) -> bool:
        """Assign a role to a user.

        Args:
            user: User to assign role to
            role_name: Role name to assign

        Returns:
            True if role was assigned, False if already assigned or not found
        """
        # Check if user already has role
        if self.user_has_role(user, role_name):
            return False

        # Get role
        role = self.db.query(Role).filter(Role.name == role_name).first()
        if not role:
            return False

        # Assign role
        user.roles.append(role)
        self.db.commit()

        return True

    def remove_role(self, user: User, role_name: str) -> bool:
        """Remove a role from a user.

        Args:
            user: User to remove role from
            role_name: Role name to remove

        Returns:
            True if role was removed, False if not found
        """
        # Find role in user's roles
        role_to_remove = None
        for role in user.roles:
            if role.name == role_name:
                role_to_remove = role
                break

        if not role_to_remove:
            return False

        # Remove role
        user.roles.remove(role_to_remove)
        self.db.commit()

        return True

    def create_role(
        self,
        name: str,
        description: str,
        permissions: List[Permission],
        is_system: bool = False,
    ) -> Role:
        """Create a new role with permissions.

        Args:
            name: Role name
            description: Role description
            permissions: List of permissions to assign
            is_system: Whether this is a system-defined role

        Returns:
            Created Role object
        """
        # Create role
        role = Role(name=name, description=description, is_system=is_system)
        self.db.add(role)
        self.db.flush()

        # Assign permissions
        for permission in permissions:
            perm = self._get_or_create_permission(permission)
            role.permissions.append(perm)

        self.db.commit()
        self.db.refresh(role)

        return role

    def _get_or_create_permission(
        self, permission: Permission
    ) -> RolePermission:
        """Get or create a permission entry.

        Args:
            permission: Permission enum value

        Returns:
            RolePermission object
        """
        perm = (
            self.db.query(RolePermission)
            .filter(RolePermission.name == permission.value)
            .first()
        )

        if not perm:
            # Parse permission name to extract resource and action
            parts = permission.value.split('_', 1)
            action = parts[0]
            resource = parts[1] if len(parts) > 1 else ''

            perm = RolePermission(
                name=permission.value,
                description=permission.name.replace('_', ' ').title(),
                resource=resource,
                action=action,
            )
            self.db.add(perm)
            self.db.flush()

        return perm

    def initialize_default_roles(self) -> None:
        """Initialize default healthcare roles with permissions.

        Should be run during application setup to create standard roles.
        """
        for role_enum, permissions in ROLE_PERMISSION_DEFAULTS.items():
            # Check if role already exists
            existing = (
                self.db.query(Role)
                .filter(Role.name == role_enum.value)
                .first()
            )

            if existing:
                continue

            # Create role description
            descriptions = {
                HealthcareRole.ADMIN: 'System administrator with full access',
                HealthcareRole.DOCTOR: 'Physician with full medical record access',
                HealthcareRole.NURSE: 'Nurse with clinical data access',
                HealthcareRole.BILLING_CLERK: 'Billing staff with financial data access',
                HealthcareRole.RECEPTIONIST: 'Front desk staff with appointment access',
                HealthcareRole.RESEARCHER: 'Researcher with de-identified data access',
                HealthcareRole.AUDITOR: 'Compliance auditor with read-only access',
            }

            self.create_role(
                name=role_enum.value,
                description=descriptions[role_enum],
                permissions=permissions,
                is_system=True,
            )
