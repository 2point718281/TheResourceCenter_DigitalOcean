import sqlite3
import shutil
import logging


class SQLDatabase:
    """
    A class to interact with an SQLite database.
    """

    def __init__(self, sql_path):
        """
        Initializes the SQLDatabase instance.

        :param sql_path: The path to the SQLite database file.
        :return: None
        """
        self.database = sqlite3.connect(sql_path, check_same_thread=False)
        self.cursor = self.database.cursor()
        self.path = sql_path
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Connected to the database at {sql_path}")

    def close(self):
        """
        Closes the database connection.

        :return: None
        """
        self.database.close()
        self.logger.info("Database connection closed.")

    def create_table(self, name, datatypes):
        """
        Creates a table with the given name and column datatypes.

        :param name: The name of the table to create.
        :param datatypes: The table column names and types.
        :return: None
        """
        self.logger.debug(
            f"Attempting to create table: {name} with datatypes: {datatypes}"
        )
        self._exec(f"CREATE TABLE {name} {datatypes};")
        self.logger.info(f"Table '{name}' created successfully.")

    def create_table_if_noexist(self, name, datatypes):
        """
        Creates a table if it does not already exist.

        :param name: The name of the table.
        :param datatypes: The table column names and types.
        :return: None
        """
        self.logger.debug(f"Checking if table '{name}' exists.")
        # Create a table if there isn't one
        self._exec(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", name
        )
        table = self.fetchone()
        if not table:
            self.logger.info(f"Table '{name}' does not exist - creating now.")
            self.create_table(name, datatypes)
        else:
            self.logger.info(f"Table '{name}' already exists - table creation halted.")

    def fetchone(self):
        """
        Fetches the next row of a query result set.

        :return: The next row of a query result set or None if no more data is available.
        """
        result = self.cursor.fetchone()
        self.logger.debug(f"Fetched one row: {result}")
        return result

    def fetchall(self):
        """
        Fetches all rows of a query result set.

        :return: A list of all rows from the result set.
        """
        result = self.cursor.fetchall()
        self.logger.debug(f"Fetched all rows: {result}")
        return result

    def _exec(self, command, *args):
        """
        Executes a SQL command.

        :param command: The SQL command to execute.
        :param args: The parameters to substitute into the command.
        :return: None
        """
        self.logger.debug(f"Executing SQL command: {command} with arguments: {args}")
        try:
            self.cursor.execute(command, args)
            self.logger.debug("SQL command executed successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"An error occurred while executing SQL command: {e}")
            raise

    def commit(self):
        """
        Saves all updates to the file

        :return: None
        """
        self.database.commit()
        self.logger.info("Database changes committed.")

    def backup(self, path):
        """
        Creates a backup of the current database.

        :param path: The path where the backup will be stored.
        :return: None
        """
        shutil.copyfile(self.path, path)
        self.logger.info(f"Database backed up to {path}")


# Set up default logging configuration
logging.basicConfig(filename='TheResourceCenter.log', level=logging.DEBUG)
