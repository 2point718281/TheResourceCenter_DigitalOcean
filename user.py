import sql
import datetime
from datetime import timezone
from enum import Enum
import secrets
import hashlib
from cryptography.fernet import Fernet
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load the encryption key from environment variable
key = os.getenv("fernet").encode()
encrypter = Fernet(key)

# Set the UTC timezone
_utc = timezone.utc

date = datetime.datetime


class Permissions(Enum):
    """Enumeration for user permissions levels."""

    ADMIN = 3
    OPERATOR = 2
    USER = 1
    LOGINFAILED = 0


class UserDatabase(sql.SQLDatabase):
    """
    UserDatabase class to manage user-related operations such as adding accounts,
    login authentication, and deleting accounts.
    Extends the SQLDatabase class to perform SQL operations on a database.
    """

    def __init__(self, sql_path):
        """
        Initialize the UserDatabase class and create the users table if it does not exist.

        :param sql_path: Path to the SQL database file.
        """
        super().__init__(sql_path)

        # Create users table if it doesn't exist
        self.create_table_if_noexist(
            "users",
            """(ID INTEGER PRIMARY KEY AUTOINCREMENT,
                DATE_CREATED DATE,
                BIRTHDAY DATE,
                USERNAME TEXT NOT NULL,
                SALT TEXT NOT NULL,
                PASSWORD TEXT NOT NULL,
                PERMISSION INTEGER,
                EMAIL TEXT NOT NULL)""",
        )
        logger.info(
            "User database initialized and users table created if not existing."
        )

    @property
    def accounts(self):
        """
        Retrieve all user accounts with their username and decrypted email.

        :return: List of tuples containing (username, email)
        """
        self._exec("SELECT USERNAME, EMAIL FROM users")
        accounts = [
            (entry[0], encrypter.decrypt(entry[1]).decode())
            for entry in self.fetchall()
        ]
        logger.debug("Fetched user accounts from database.")
        return accounts

    @property
    def all_information(self):
        """
        Retrieve all information from the users table.

        :return: List of tuples containing all user information.
        """
        self._exec("SELECT * FROM users")
        logger.debug("Fetched all user information from database.")
        return self.fetchall()

    def login(self, username, password):
        """
        Attempt to log in a user with the provided username and password.

        :param username: Username of the account.
        :param password: Password of the account.
        :return: Permissions level if login is successful, otherwise LOGINFAILED.
        """
        logger.info("Attempting login for user: %s", username)
        if username in [entry[0] for entry in self.accounts]:
            self._exec(
                "SELECT ID, USERNAME, PASSWORD, PERMISSION FROM users WHERE USERNAME=?",
                username,
            )
            for result in self.fetchall():
                salt = self.get_salt(result[0])
                hashed_password = hashlib.sha256((password + salt).encode()).hexdigest()
                if username == result[1] and result[2] == hashed_password:
                    perms = result[3]
                    for permname in Permissions.__members__:
                        permlevel = Permissions.__members__[permname]
                        if permlevel.value == perms:
                            logger.info(
                                "Login successful for user: %s, with permision level %s",
                                username,
                                permname,
                            )
                            return permlevel

        logger.warning("Login failed for user: %s", username)
        return Permissions.LOGINFAILED

    def add_account(
        self, birthday, username, password, email, perms=Permissions.USER, creation=None
    ):
        """
        Add a new user account to the database.

        :param birthday: User's birthday.
        :param username: Username for the new account.
        :param password: Password for the new account.
        :param email: Email address for the new account.
        :param perms: Permission level for the user (default is USER).
        :param creation: Creation date (defaults to current date if not provided).
        :return: ID of the newly created account.
        """
        logger.info("Adding new account for user: %s", username)
        if not creation:
            creation = str(datetime.datetime.now(_utc)).split(" ")[0]

        birthday = str(birthday).split(" ")[0]
        id_ = self.get_last_id() + 1
        salt = self.get_salt(id_)
        email = encrypter.encrypt(email.encode())
        hashed_password = hashlib.sha256((password + salt).encode()).hexdigest()
        self._exec(
            "INSERT INTO users (DATE_CREATED, BIRTHDAY, USERNAME, SALT, PASSWORD, PERMISSION, EMAIL) VALUES (?, ?, ?, ?, ?, ?, ?);",
            creation,
            birthday,
            username,
            salt,
            hashed_password,
            int(perms.value),
            email,
        )
        self.commit()
        logger.info("Account created for user: %s", username)
        return id_

    def drop_user(self, id_=None, username=None):
        """
        Delete a user from the database by ID or username.

        :param id_: ID of the user to delete.
        :param username: Username of the user to delete.
        """
        if id_:
            self._exec("DELETE FROM users WHERE ID=?", id_)
            logger.info("Deleted user with ID: %d", id_)
        elif username:
            self._exec("DELETE FROM users WHERE USERNAME=?", username)
            logger.info("Deleted user with username: %s", username)

    def get_salt(self, id_):
        """
        Retrieve or generate a salt for the given user ID.

        :param id_: ID of the user.
        :return: Salt as a hexadecimal string.
        """
        if (
            id_ <= self.get_last_id()
        ):  # We're looking for a salt from an existing account
            try:
                self._exec("SELECT ID, SALT FROM users WHERE ID=?", id_)
                salt = self.fetchone()[1]
                logger.debug("Retrieved salt for user ID: %d", id_)
                return salt
            except Exception as e:
                logger.error(
                    "Error retrieving salt for user ID: %d. Generating new salt. Error: %s",
                    id_,
                    e,
                )
                pass

        # We're generating a new salt
        salt = secrets.token_bytes(16)
        logger.debug("Generated new salt for user ID: %d", id_)
        return hashlib.sha256(salt).hexdigest()

    def get_last_id(self):
        """
        Retrieve the last user ID in the users table.

        :return: Last user ID or 0 if the table is empty.
        """
        self._exec("SELECT ID FROM users")
        all_ids = self.fetchall()
        if all_ids:
            logger.debug("Retrieved last user ID from database.")
            return all_ids[-1][0]

        logger.debug("No users found in database. Returning ID 0.")
        return 0

    def __len__(self):
        """
        Retrieve the number of users in the users table.

        :return: Number of users in the database.
        """
        self._exec("SELECT ID FROM users")
        count = len(self.fetchall())
        logger.debug("Number of users in database: %d", count)
        return count


# Instantiate the UserDatabase with a database file path
users = UserDatabase("users.db")
