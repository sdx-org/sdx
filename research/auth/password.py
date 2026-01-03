"""Password hashing and verification utilities.

Implements secure password handling following HIPAA Technical Safeguards
requirements for password management.
"""

from passlib.context import CryptContext
from passwordlib.commonly_used import is_commonly_used
# Use bcrypt for password hashing (HIPAA compliant)
# Work factor of 12 provides strong security while maintaining performance
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
def contain_special_chars(input_string :str)->bool:
    return any(not c.isalnum() for c in input_string)
## NOTE: Instead of a class we may keep these in utils as
## NOTE: independent functions as they don't share a state in specific.
class PasswordManager:
    """Manages password hashing, verification, and strength validation.

    This class provides a convenient interface for password operations
    while maintaining HIPAA-compliant security best practices.

    Attributes:
        plain_password: The plain text password to be processed.
        hashed_password: Optional pre-hashed password for verification.
    """
    @staticmethod
    def get_password_hash(plain_password: str) -> str:
        """Hash the plain password for secure storage.

        Returns:
            str: Hashed password suitable for database storage.
        """
        return pwd_context.hash(plain_password)

    @staticmethod
    def verify_password(plain_password: str,hashed_password: str) -> bool:
        """Verify the plain password against a hashed password.

        Args:
            plain_password: password string by the user.
            hashed_password: Hashed password to compare against.

        Returns:
            bool: True if password matches the hash, False otherwise.

        Raises:
            ValueError: If no hashed password is provided or stored.
        """
        if not hashed_password:
            raise ValueError('No hashed password provided for verification')
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, list[str]]:
        """Validate that a password meets security requirements.

        HIPAA requires strong password policies including:
        - Minimum length of 12 characters
        - Complexity requirements (upper, lower, digit, special char)
        - No common password patterns

        Args:
            password: Password string to validate.

        Returns:
            tuple[bool, list[str]]: A tuple containing:
                - bool: True if password is valid, False otherwise.
                - list[str]: List of error messages if invalid, empty list if valid.
        """
        errors = []
        if len(password) < 12:
            errors.append('Password must be at least 12 characters long')

        # Must contain uppercase letter, lowercase letter and digits
        if not any(c.isupper() for c in password):
            errors.append('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in password):
            errors.append('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in password):
            errors.append('Password must contain at least one digit')
        if not contain_special_chars(password):
            errors.append('Password must contain at least one special character')
        if is_commonly_used(password):
            errors.append('Password is too common')
        return len(errors) == 0, errors
