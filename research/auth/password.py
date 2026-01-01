"""Password hashing and verification utilities.

Implements secure password handling following HIPAA Technical Safeguards
requirements for password management.
"""

from passlib.context import CryptContext

# Use bcrypt for password hashing (HIPAA compliant)
# Work factor of 12 provides strong security while maintaining performance
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

# Special characters for password validation
SPECIAL_CHARS = '!@#$%^&*()_+-=[]{}|;:,.<>?'


class PasswordManager:
    """Manages password hashing, verification, and strength validation.

    This class provides a convenient interface for password operations
    while maintaining HIPAA-compliant security best practices.

    Attributes:
        plain_password: The plain text password to be processed.
        hashed_password: Optional pre-hashed password for verification.
    """
    # TODO: Make this an env varibale(comma-separated list)
    COMMON_PASSWORDS = [
        'password',
        'password123',
        'admin123',
        'healthcare',
        'medical123',
    ]

    def __init__(
        self, plain_password: str, hashed_password: str = None
    ) -> None:
        """Initialize the PasswordManager.

        Args:
            plain_password: Plain text password to be hashed or verified.
            hashed_password: Optional hashed password for verification operations.
        """
        self.plain_password = plain_password
        self.hashed_password = hashed_password

    def get_password_hash(self) -> str:
        """Hash the plain password for secure storage.

        Returns:
            str: Hashed password suitable for database storage.
        """
        return pwd_context.hash(self.plain_password)

    def verify_password(self, hashed_password: str = None) -> bool:
        """Verify the plain password against a hashed password.

        Args:
            hashed_password: Hashed password to compare against. If not provided,
                uses the instance's hashed_password attribute.

        Returns:
            bool: True if password matches the hash, False otherwise.

        Raises:
            ValueError: If no hashed password is provided or stored.
        """
        hash_to_verify = hashed_password or self.hashed_password
        if hash_to_verify is None:
            raise ValueError('No hashed password provided for verification')
        return pwd_context.verify(self.plain_password, hash_to_verify)

    def validate_strength(self) -> tuple[bool, list[str]]:
        """Validate that the password meets security requirements.

        Checks HIPAA-compliant password policy including:
        - Minimum length of 12 characters
        - Uppercase and lowercase letters
        - At least one digit
        - At least one special character
        - Not a common password

        Returns:
            tuple[bool, list[str]]: A tuple containing:
                - bool: True if password is valid, False otherwise.
                - list[str]: List of error messages if invalid, empty if valid.
        """
        return self.validate_password_strength(self.plain_password)

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

        # Minimum length of 12 characters (HIPAA best practice)
        if len(password) < 12:
            errors.append('Password must be at least 12 characters long')

        # Must contain uppercase letter
        if not any(c.isupper() for c in password):
            errors.append(
                'Password must contain at least one uppercase letter'
            )

        # Must contain lowercase letter
        if not any(c.islower() for c in password):
            errors.append(
                'Password must contain at least one lowercase letter'
            )

        # Must contain digit
        if not any(c.isdigit() for c in password):
            errors.append('Password must contain at least one digit')

        # Must contain special character
        if not any(c in SPECIAL_CHARS for c in password):
            errors.append(
                'Password must contain at least one special character'
            )

        if password.lower() in PasswordManager.COMMON_PASSWORDS:
            errors.append('Password is too common')

        return len(errors) == 0, errors
