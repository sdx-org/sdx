"""Mappings for Role-Based Access Control (RBAC).

This module defines default permission mappings for healthcare roles,
implementing the principle of least privilege and minimum necessary access
as required by HIPAA.
"""

from typing import Final

from research.models.rbac import HealthcareRole, Permission

# Type alias for role-permission mapping
RolePermissionMapping = dict[HealthcareRole, list[Permission]]

ROLE_PERMISSION_DEFAULTS: Final[RolePermissionMapping] = {
    HealthcareRole.ADMIN: [
        Permission.VIEW_PATIENT_DEMOGRAPHICS,
        Permission.EDIT_PATIENT_DEMOGRAPHICS,
        Permission.MANAGE_USERS,
        Permission.MANAGE_ROLES,
        Permission.VIEW_AUDIT_LOGS,
        Permission.VIEW_APPOINTMENTS,
        Permission.MANAGE_APPOINTMENTS,
    ],
    HealthcareRole.DOCTOR: [
        Permission.VIEW_PATIENT_DEMOGRAPHICS,
        Permission.EDIT_PATIENT_DEMOGRAPHICS,
        Permission.VIEW_MEDICAL_RECORDS,
        Permission.EDIT_MEDICAL_RECORDS,
        Permission.VIEW_DIAGNOSIS,
        Permission.EDIT_DIAGNOSIS,
        Permission.VIEW_PRESCRIPTIONS,
        Permission.EDIT_PRESCRIPTIONS,
        Permission.VIEW_LAB_RESULTS,
        Permission.EDIT_LAB_RESULTS,
        Permission.VIEW_MENTAL_HEALTH,
        Permission.EDIT_MENTAL_HEALTH,
        Permission.VIEW_APPOINTMENTS,
        Permission.MANAGE_APPOINTMENTS,
    ],
    HealthcareRole.NURSE: [
        Permission.VIEW_PATIENT_DEMOGRAPHICS,
        Permission.VIEW_MEDICAL_RECORDS,
        Permission.EDIT_MEDICAL_RECORDS,
        Permission.VIEW_DIAGNOSIS,
        Permission.VIEW_PRESCRIPTIONS,
        Permission.VIEW_LAB_RESULTS,
        Permission.EDIT_LAB_RESULTS,
        Permission.VIEW_APPOINTMENTS,
    ],
    HealthcareRole.BILLING_CLERK: [
        Permission.VIEW_PATIENT_DEMOGRAPHICS,  # Minimum necessary for billing
        Permission.VIEW_BILLING,
        Permission.EDIT_BILLING,
        Permission.PROCESS_PAYMENTS,
    ],
    HealthcareRole.RECEPTIONIST: [
        Permission.VIEW_PATIENT_DEMOGRAPHICS,  # Name, contact info only
        Permission.VIEW_APPOINTMENTS,
        Permission.MANAGE_APPOINTMENTS,
    ],
    HealthcareRole.RESEARCHER: [
        Permission.VIEW_DEIDENTIFIED_DATA,
        Permission.EXPORT_RESEARCH_DATA,
    ],
    HealthcareRole.AUDITOR: [
        Permission.VIEW_AUDIT_LOGS,
        Permission.VIEW_PATIENT_DEMOGRAPHICS,
        Permission.VIEW_MEDICAL_RECORDS,
        Permission.EXPORT_DATA,
    ],
}
