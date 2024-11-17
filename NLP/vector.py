import os
import logging
import json

logging.basicConfig(
    filename="TheResourceCenter.log", encoding="utf-8", level=logging.DEBUG
)
logger = logging.getLogger(__name__)
logger.info("LOGGER ACTIVATED")

try:
    with open(os.path.join("..", "sql.py")) as f:
        exec(f.read(), globals())

except:
    with open("sql.py") as f:
        exec(f.read(), globals())


class VectorDatabase(SQLDatabase):
    """A database used to efficiently store the weights for words"""

    def __init__(self, path, dim):
        super().__init__(path)

        # Initialization
        self.dim = dim
        self.types = (
            "(WORD TEXT,"
            + ", ".join([f"W{i} DOUBLE PRECISION" for i in range(dim)])
            + ")"
        )
        self.forinsert = "(WORD, " + ", ".join([f"W{i}" for i in range(dim)]) + ")"

        self.default = [0] * dim

        self.create_table_if_noexist("words", self.types)

    def add_word(self, word, weights, autocommit=True):
        """
        Adds a word to the database

        :param word: The word to add to the database
        :param weights: The weights of the word embedding as a list
        """
        # Parse the parameters into SQL commands.

        # Forinsert is not user-inputted so we can just plug it in with an fstring instead of using question marks.
        # Execute the SQL command.
        params = [word] + weights
        self._exec(
            f"INSERT INTO words {self.forinsert} VALUES ("
            + ", ".join(["?" for _ in range(self.dim + 1)])
            + ");",
            *params,
        )

        if autocommit:
            self.commit()  # Save the database if requested

    def get_word(self, word):
        """
        Gets the weights for a word

        :param word: Word to get the weights of.
        """
        self._exec("SELECT * FROM words WHERE WORD=?;", word)
        for row in self.cursor:  # Iterate through all words so we read them one by one, thereby lowering the memory cost of this function.
            if row[0] == word.lower():
                return list(row[1:])

        return self.default  # Return the default value if we find no results.

    def __setitem__(self, word, weights):
        """
        Alias for "add_word"
        """
        self.add_word(self, word, weights)

    def __getitem__(self, word):
        """
        Alias for "get_word"
        """
        return self.get_word(word)


def compile_semantics(filename, source, words="all", format_="sql"):
    """
    Compile semantic vectors for given words using a pre-trained GloVe embedding.

    :param filename: Path to the output file where semantics will be stored.
    :param words: List of words for which semantic vectors are needed, or the string "all" to fetch all available vectors
    :return: None
    """
    logger.info("Compiling semantic vectors")
    semantics = {}
    try:
        # Read the pre-trained GloVe embeddings
        with open(source, "rb") as f:
            for line in f.read().decode().split("\n"):
                try:
                    word_data = line.split()
                    semantics[word_data[0]] = [float(i) for i in word_data[1:]]
                except Exception:
                    logger.error(f'Could not get semantic information for "{line}"')
    except FileNotFoundError as e:
        logger.error(f"Error opening GloVe file: {e}")
        return

    if isinstance(words, str) and words.lower() == "all":
        words = [word.lower() for word in semantics]

    if format_ == "json" or filename == "return":
        default = [0] * 50
        dictionary = {word: semantics.get(word, default) for word in words}

        if filename != "return":
            with open(filename, "w") as f:
                f.write(json.dumps(dictionary))

        else:
            return dictionary

    elif format_ == "sql":
        database = VectorDatabase(filename, 50)

        for word in words:
            database.add_word(word, semantics[word], False)

        database.commit()


vectors = VectorDatabase(os.path.join(os.path.dirname(__file__), "semantics.db"), 50)
