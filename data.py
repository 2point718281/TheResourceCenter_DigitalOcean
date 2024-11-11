# Imports
import csv
import logging
import string
import requests
import os
import datetime
import nltk
import numpy as np
from NLP.vector import compile_semantics
import json

# Download necessary NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")

# Set up logging for debugging
logging.basicConfig(encoding="utf-8", level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info("LOGGER ACTIVATED")


def split(string):
    return string.split()


# CSV Filename for database
broad_filename = "Curated_DatabaseV4.csv"
broad_subject_mapping_file = "broadsubjectmappings.json"

with open(os.path.join("static", "data", broad_subject_mapping_file)) as f:
    mappings = json.loads(f.read())
    broad_subject_mapping = mappings["broad"]
    rtypes = mappings["rtypes"]


def correct_excel(file):
    """
    Remove redundant commas from a CSV file.

    :param file: Path to the CSV file to be corrected.
    :return: None
    """
    logger.info(f"Correcting redundant commas in file: {file}")
    with open(file) as f:
        txt = f.read()

    last_txt = ""

    # Loop until all redundant commas are removed
    while txt != last_txt:
        last_txt = txt
        txt = txt.replace("/ ", "/")
        logger.debug(f"Correcting slashes in file {file}")

    last_txt = ""

    while txt != last_txt:
        last_txt = txt
        txt = txt.replace(",,", ",")
        logger.debug(f"Correcting redundant commas in file {file}")

    # Save the corrected content back to the file
    with open(file, "w") as f:
        f.write(txt)
    logger.info(f"File {file} corrected successfully")


correct_excel(os.path.join("static", "data", broad_filename))


def remove_punctuation(s):
    """
    Remove all punctuation from the input string.

    :param s: Input string to be processed.
    :return: String with all punctuation removed.
    """
    logger.debug(f"Removing punctuation from string: {s}")
    return s.translate(str.maketrans("", "", string.punctuation))


def checklinks(file, link_index):
    """
    Check the validity of URLs in a specified column of a CSV file.

    :param file: Path to the CSV file containing URLs.
    :param link_index: Index of the column containing the URLs to be checked.
    :return: None
    """
    logger.info(f"Checking links in file {file}")
    with open(file, newline="") as f:
        reader = csv.reader(f)
        data = [row for row in reader][1:]  # Skip header row

    # Iterate through each row to verify the URL
    for x, i in enumerate(data):
        url = i[link_index]
        try:
            response = requests.get(url)
            if response.status_code == 404:
                logger.warning(f"Row {x + 1}: URL {url} returned 404 Not Found")
            else:
                logger.info(
                    f"Row {x + 1}: URL {url} is valid with status code {response.status_code}"
                )
        except requests.RequestException as e:
            logger.error(f"Row {x + 1}: Error occurred while checking URL {url}: {e}")


def excel_serial_to_date(serial):
    """
    Convert an Excel serial date to a readable date format.

    :param serial: Excel serial date to be converted.
    :return: Converted date in readable format.
    """
    logger.debug(f"Converting Excel serial date {serial}")
    base_date = datetime.datetime(
        1899, 12, 30
    )  # Excel's date system starts from 1899-12-30
    return base_date + datetime.timedelta(days=int(serial))


def determine_approx_rating(qlen):
    """
    Determines the minimum required rating for a result to be considered a "Good match" for a query of a certain length

    :param qlen: Length of the query in tokens (words)
    :return: Approximate rating a result should have for it to be considered "good"
    """
    m, b = (
        9.7443125,
        -6.612125,
    )  # Use the precomputed linear regression to find the required rating.
    return m * qlen + b


def rating_to_word(qlen, rating):
    """
    DEPRECIATED, Reason: Did not serve function properly, replaced by "position_as_word"
    Converts the rating to a human-readable word, i.e. "1000.23424" to "Best", and "0" to "Irrelevant"

    :param qlen: Length of the query in tokens (words)
    :param rating: The approximate rating of the result as a number.
    :return: Approximate rating of the result as a word
    """
    minimum = determine_approx_rating(qlen)  # Determine the minimum rating we need

    if rating < minimum:
        return "Irrelevant"  # If the rating is below the minimum then this result is irrelevant.

    additions = list(
        {
            "Okay": 0,  # The amount by which the rating has to surpass the minimum for it to be considered each level of Good
            "Good": 3,
            "Better": 6,
            "Best": 10,
        }.items()
    )

    for x, r in enumerate(additions):
        if rating > minimum + r[1]:
            if x + 1 != len(additions):
                if rating < minimum + additions[x + 1][1]:
                    return r[0]

            else:
                return r[0]


def position_as_word(position, total):
    """
    Gives a human readable rating (Like "Best", "Better", or "Good") given a result's position relative to other results

    :param position: The result's ranking in the list of resources (position should be the number of results that are said to be "Better")
    :param total: The total number of results being considered
    :return: Approximate rating of the result as a word
    """
    if position <= 10 / 100 * total:  # First 10% are the Best results
        return "Best"

    elif position <= 30 / 100 * total:  # Next 10% are Better results
        return "Better"

    elif position <= 80 / 100 * total:  # Next 50% are Good results
        return "Good"

    else:  # Other results are okay
        return "Okay"


class BroadDatabase:
    """
    Class for managing and querying a broad database stored in a CSV file.
    """

    def __init__(self, file):
        """
        Initialize the BroadDatabase instance by loading data from a CSV file.

        :param file: Path to the CSV file to be loaded.
        :return: None
        """
        logger.info(f"Initializing BroadDatabase with file {file}")
        self.file = file

        try:
            # Load CSV data
            with open(file, newline="\n") as f:
                reader = csv.reader(f)
                data = [row for row in reader]
                self.focused_titles = [i.strip().replace("/ ", "/") for i in data[0]][
                    :-1
                ]  # The titles are in the first row
                self.titles = self.focused_titles[:10]
                self.data = [
                    row[:-1] for row in data[1:]
                ]  # All the other rows are data
        except FileNotFoundError as e:
            logger.error(f"Error loading CSV file: {e}")
            return

        self.semantic = compile_semantics(
            "return",
            os.path.join("static", "semantics", "glove.6B.50d.txt"),
            words="all",
            format_="json",
        )

        self.clean_data()  # Clean up the data
        self.words = self.extract_words()  # Extract words for TF-IDF
        self.relevant = (
            0,
            1,
            2,
            3,
            6,
            8,
            9,
        )  # Indices of relevant columns that should be shown to the user
        self.types = (
            str,
            str,
            list,
            str,
            str,
            str,
            str,
            str,
            str,
            str,
            list,
        )  # Expected data types for each column
        self.idfs = {
            i: self.words.count(i) for i in set(self.words)
        }  # Calculate inverse document frequency for words
        self.titles.append("")
        logger.info(
            f"BroadDatabase initialized successfully with {len(self.data)} rows of data"
        )

    def clean_data(self):
        """
        Clean the data by processing age and grade fields and filtering out problematic rows.
        Add semantic information to each row.

        :return: None
        """
        logger.info("Cleaning data")
        cleaned_data = []
        for i in self.data:
            try:
                # Process the subject field
                i[2] = i[2].split(", ")

                # Process the age range field by extracting numbers and creating an appropriate range
                range_ = [
                    int(val)
                    for val in [
                        n
                        for n in "".join(
                            [ch for ch in i[3] if ch.isdigit() or ch == "-"]
                        ).split("-")
                        if n
                    ]
                ]
                i[3] = i[3].strip('"')

                try:
                    i[10] = int(i[10])
                    i[11] = int(i[11])

                except Exception:
                    pass

                try:
                    subjects = {}
                    for broad_subject in i[14].split(";"):
                        split = broad_subject.split("-")
                        subjects[split[0].strip()] = [
                            sub_subject.strip() for sub_subject in split[1].split(",")
                        ]
                    i[14] = subjects

                except Exception:
                    pass

                # Add semantic vector for the entire row by summing the vectors for each word
                words = " ".join(
                    [
                        str(element)
                        if isinstance(element, (str, int))
                        else " ".join([str(e) for e in element])
                        for element in i
                    ]
                )
                words = words.split()
                ls = [self.semantic.get(word, [0] * 50) for word in words]
                i.append(np.array(ls).sum(axis=0).tolist())
                cleaned_data.append(i)
                logger.debug(f"Row cleaned successfully: {i}")
            except Exception as e:
                # Log any errors that occur during data cleaning
                logger.warning(f"Deleting row {i} due to error: {e}")
        self.focused_data = cleaned_data
        self.data = [row[:10] + [row[-1]] for row in self.focused_data]
        logger.info(
            f"Data cleaning completed. Cleaned data contains {len(self.data)} rows"
        )

    def extract_words(self):
        """
        Extract all words from the dataset for further analysis.

        :return: List of words extracted from the dataset.
        """
        logger.info("Extracting words from data")
        words = []
        for i in self.data:
            for j in i:
                if isinstance(j, list):
                    if isinstance(j[0], (int, float)):
                        continue
                    words.extend([n.lower() for n in j])
                else:
                    # Remove punctuation and convert string to lowercase
                    cleaned_str = remove_punctuation(j).lower()
                    words.extend(cleaned_str.split())
        logger.info(f"Extracted {len(words)} words from the dataset")
        return words

    def determine_tfidf(self, string_, id_or_contest):
        """
        Determine the TF-IDF score of a given search query for a particular contest.

        :param string_: Search query string.
        :param id_or_contest: Either the index of a contest or a list representing a contest.
        :return: The TF-IDF score for the given query.
        """
        logger.debug(f"Calculating TF-IDF for query: {string_}")
        # Remove punctuation and stopwords from the search query
        string_ = remove_punctuation(string_).lower()
        string_ = [
            word
            for word in string_.split()
            if word not in nltk.corpus.stopwords.words("english")
        ]

        # Determine which dataset to compare against - either by index or by direct input
        if isinstance(id_or_contest, int):
            test = [
                elem.lower()
                if isinstance(elem, str)
                else [sub_elem.lower() for sub_elem in elem]
                for elem in self.data[id_or_contest][:-2]
            ]
        else:
            test = [
                elem.lower()
                if isinstance(elem, str)
                else [sub_elem.lower() for sub_elem in elem]
                for elem in id_or_contest[:-2]
            ]

        test = [_ for _ in test if _]

        # Calculate the TF-IDF score for each word in the query
        tfidf = 0
        for word in string_:
            tfidf += max(
                [
                    self.idfs.get(word, 1)
                    * element.count(word)
                    / len(element)
                    * len(word)
                    if isinstance(element, str)
                    else self.idfs.get(word, 1)
                    * max([sub_elem.count(word) for sub_elem in element])
                    / len("".join(element))
                    * len(word)
                    for element in test
                ]
            )

        logger.debug(f"TF-IDF score for query '{string_}': {tfidf}")
        return tfidf

    def determine_semantic(self, string, result):
        """
        Determine the semantic similarity between a search query and a given result.

        :param string: Search query string.
        :param result: The contest details.
        :return: The semantic similarity score.
        """
        # Convert the semantic vectors to numpy arrays
        semantic_r = np.array(result[-1])
        ls = [self.semantic.get(word, [0] * 50) for word in string.split()]
        semantic_s = np.array(ls).sum(axis=0)

        # Calculate cosine similarity between the search vector and the result vector
        similarity_score = (
            np.dot(semantic_r, semantic_s)
            / (np.linalg.norm(semantic_r) * np.linalg.norm(semantic_s))
            + 1
        )
        logger.debug(
            f"Semantic similarity score for query '{string}': {similarity_score}"
        )
        return similarity_score

    def determine_tfidfstring(self, string, x):
        words = split(string)
        return sum(
            [
                self.determine_tfidf(word, x) * self.determine_semantic(word, x)
                for word in words
            ]
        ) / len(words)

    def _search(self, string="", subset=None, raw=False):
        """
        Internal search function to rank results based on query match.

        :param string: Search query.
        :param subset: Subset of data to search in.
        :return: List of search results based on query match.
        """
        logger.info(f"Performing internal search for query: '{string}'")
        if not subset:
            subset = self.data.copy()

        # Calculate scores for each entry in the subset using TF-IDF and semantic similarity
        qlen = len(split(string))
        required_score = determine_approx_rating(qlen)
        scores = [self.determine_tfidfstring(string, x) for x in subset]
        # Sort the results by score in descending order and return those that match the threshold
        search_results = [
            i
            for i in sorted(zip(scores, subset), key=lambda x: x[0], reverse=True)
            if i[0] > required_score
        ]

        if raw:
            res = [
                (
                    (i[0], i[1][:-1] + [search_results[x][0], i[1][-1]]),
                    position_as_word(x, len(search_results)),
                )
                for x, i in enumerate(search_results)
            ]

        else:
            res = [
                (i, position_as_word(x, len(search_results)))
                for x, i in enumerate(search_results)
            ]
        logger.info(f"Search completed. Found {len(search_results)} matching results")
        return res

    def convert(self, result, raw, rating=None):
        """
        Convert a search result into a readable format.

        :param result: Search result to be converted.
        :return: Converted result in a readable format.
        """
        logger.debug(f"Converting result: {result}")
        result = result[:-1]
        r = [
            (self.titles[i], result[i])
            if self.types[i] == str
            else (self.titles[i], str(result[i])[1:-1].replace("'", ""))
            for i in self.relevant
        ]
        if raw:
            r.insert(-1, ("Raw Rating", result[10]))
        # Process the age range to ensure correct formatting in the output

        if rating is not None:
            r.insert(-1, ("Match Level", rating))

        logger.debug(f"Converted result: {r}")
        return r

    def search(self, string, raw=False, num_res=100):
        """
        Public search function to search the database. This controls Broad Search.

        :param string: Search query.
        :param num_res: Number of results to return.
        :return: List of search results.
        """
        logger.info(
            f"Searching database for query: '{string}' with max results {num_res}"
        )
        filtered = self._search(string, None, raw)
        # Convert results to a human-readable format and return
        results = [self.convert(i[0][1], raw, i[1]) for i in filtered[:num_res]]
        logger.info(f"Search completed. Returning {len(results)} results")
        return results

    def search_specific(self, query, ages, subjects_, rtypes, raw=False):
        """
        Search function with specific filters for ages, subjects, and types. This controls Focused Search.

        :param query: Search query.
        :param ages: set object listing the ages selected by the user
        :param subjects_: List of subject filters, usually from the json file "broadsubjectmappings.json"
        :param rtype: List of resource types the user has selected.
        :param raw: Add the raw tf-idf and semantic rating to the result for revision
        :return: List of filtered search results.
        """
        if isinstance(rtypes, str):
            rtypes = [rtypes]

        logger.info(
            f"Performing specific search with query: '{query}', ages: {ages}, subjects: {subjects_}, resource types: {rtypes}"
        )
        # Create a set of subject categories to match against, and match each result against them
        all_subjects = subjects_.union(
            *[broad_subject_mapping.get(subject, set()) for subject in subjects_]
        )

        def get_relevant_subjects(result):
            return set(result[14].keys()).union(
                *[set(result[14].get(key, set())) for key in result[14]]
            )

        # Match by subject and resource type
        filtered = [
            i
            for i in self.focused_data
            if get_relevant_subjects(i).intersection(all_subjects)
        ]

        filtered = [
            i
            for i in filtered
            if any(
                [
                    i[15].lower().strip().replace("/ ", "/").startswith(rtype.lower())
                    for rtype in rtypes
                ]
            )
        ]

        # Filter the data based on age range

        filtered = [
            i
            for i in filtered
            if any([i[10] <= age_option <= i[11] for age_option in ages])
        ]
        # Finally, filter it based on the query
        if filtered:
            results = []
            if query:
                for i in self._search(
                    query, [row[:10] + [row[-1]] for row in filtered], raw=raw
                ):
                    try:
                        results.append(self.convert(i, raw=raw))
                    except Exception as e:
                        logger.error(f"Error converting result {i}: {e}")

            else:
                for row in filtered:
                    results.append(self.convert(row[:10] + [row[-1]], raw=False))

            logger.info(f"Specific search completed. Found {len(results)} results")
            return results

        else:
            logger.info("Specific search completed. Found 0 results")
            return []


# Instantiate the BroadDatabase object

broad = BroadDatabase(os.path.join("static", "data", broad_filename))


# Uncomment to identify semantic information
# print('Compiling semantics...')
# compile_semantics('semantics.json', os.path.join('static', 'semantics', 'glove.6B.50d.txt'), words='all', format_='json')
# print('Done')
