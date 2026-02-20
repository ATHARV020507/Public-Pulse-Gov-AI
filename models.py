from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# 1. Initialize the Database Extension
db = SQLAlchemy()

# 2. Define the User Table
class User(UserMixin, db.Model):
    """
    Database table to store Officer credentials securely.
    Inherits from UserMixin to work with Flask-Login.
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False) # Stores HASH, not real password

    def set_password(self, password):
        """
        Takes a plain password (e.g., 'sih2025') and converts it 
        to a secure cryptographic hash.
        """
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """
        Checks if the provided password matches the stored hash.
        Returns True if correct, False if wrong.
        """
        return check_password_hash(self.password_hash, password)