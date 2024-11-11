# imports
import csv
import logging
import difflib
import string
import find_dates
import os
import requests
import nltk

contest_filename = "Contests_DataBase.csv"
freeresources_filename = "Free_Educational_Resources.csv"
awards_filename = "Awards_Scholarships_Bursaries_Database.csv"
broad_filename = "Curated_Database.csv"

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


def correct_excel(file):
    with open(file) as f:
        txt = f.read()

    last_txt = ""

    while txt != last_txt:
        last_txt = txt
        txt = txt.replace(",,", ",")

    with open(file, "w") as f:
        f.write(txt)


# global variables
GENERAL_SUBJECTS = {
    "math": (
        "math",
        "alpha",
        "theta ",
        " theta",
        " pi",
        "pi ",
        " mandelbrot",
        "mandelbrot ",
    ),
    "science": ("physics", "biology", "science", "bio", "genetic", "chem"),
    "coding": ("computing", "coding"),
    "robotics": ("robotics", "engineer", "robo"),
    "geography": ("geography",),
    "writing": (
        "poet",
        "lang",
        "write",
        "writing",
    ),
    "history": ("history",),
    "nature": ("nature",),
    "film": ("film",),
}  # you shouldn't need to edit these, but if you do edit results.js too

SPECIFIC_SUBJECTS = {}  # use this for exceptions (contests that have unusual names)

SUBJECTS = {key: tuple() for key in list(GENERAL_SUBJECTS) + list(SPECIFIC_SUBJECTS)}

for key in SUBJECTS:
    SUBJECTS[key] = GENERAL_SUBJECTS.get(key, tuple()) + SPECIFIC_SUBJECTS.get(
        key, tuple()
    )

# set up logging for debugging
logger = logging.getLogger(__file__)
logging.basicConfig(filename="contestlist.log", encoding="utf-8", level=logging.DEBUG)
logging.info("LOGGER ACTIVATED")

# this matcher is used to check the similarity between two strings. It's used for the search function.
matcher = difflib.SequenceMatcher()


def remove_punctuation(s):
    """Removes all punctuation from the input string"""
    for value in string.punctuation:
        s = s.replace(value, "")  # replace all the punctuation with blank strings

    return s


def tokenize(string, split=False):
    string = remove_punctuation(string).lower()
    if split:
        return string.split()
    return list(string.strip())


def similarity(s1, s2):
    s1t, s2t = tokenize(s1), tokenize(s2)
    s2t.copy()

    num_overlapl = 0
    for token in s1t:
        if token in s2t:
            s2t.remove(token)
            num_overlapl += 1

    s1t, s2t = tokenize(s1, True), tokenize(s2, True)
    num_overlap = 0
    for token in s1t:
        if token in s2t:
            s2t.remove(token)
            num_overlap += 1

    return num_overlap / len(s1) + 0.1 * num_overlapl / len(s1t) + 0.01 * ratio(s1, s2)


def ratio(s1, s2):
    """Calculates the similarity ratio between two strings. This function is oriented to specialize in searching for strings"""
    matcher.set_seq1(s1)
    matcher.set_seq2(s2)
    return matcher.ratio()


class ContestList:
    # this variable is used to translate from human grades to numbers
    GRADES = {
        "k": 0,
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "11": 11,
        "12": 12,
        "99": 99,
    }

    def __init__(self, file):
        """Initializes the contest list"""
        # read the file
        self.file = file
        with open(file, newline="") as f:
            reader = csv.reader(f)

            data = [
                row[:8] for row in reader
            ]  # the database contains some extra characteristics we don't need. Trim it to only the first 8 columns
            self.titles = data[0]  # the titles are in the first row
            self.data = data[1:]  # all the other rows are data

        # translate strings like "Grades 9-12" to a tuple like (9, 12)
        for row in self.data:
            grades = row[6]  # fetch the value
            grades = (
                grades.lower()
                .replace("grades", "")
                .replace("grade", "")
                .replace(" ", "")
            )  # remove unnecessary information
            grades = grades.split("-")  # split it along the dash

            if (
                len(grades) == 1
            ):  # The format at the start could have been "Grade 9", in which case the tuple would be (9, 9)
                row[6] = (ContestList.GRADES[grades[0]], ContestList.GRADES[grades[0]])

            elif (
                len(grades) == 2
            ):  # the format could have been "Grades 9-12", in which case the tuple would be (9, 12)
                row[6] = (ContestList.GRADES[grades[0]], ContestList.GRADES[grades[1]])

        # determine the subjects of each contest.
        for contest in self.data.copy():
            relevant = set()  # keep track of the relevant subjects
            for (
                subject,
                s,
            ) in SUBJECTS.items():  # for each subject, loop through its key phrases
                for s_ in s:
                    if (
                        s_.lower() in contest[0].lower()
                        or s_.lower() in contest[5].lower()
                    ):  # if the contest's name contains any key phrases, add the subject to the relevant list
                        relevant.add(subject)
            contest.append(relevant)

        self.relevant = 0, 1, 2, 3, 4, 5, 6
        self.data.sort(key=lambda x: x[0][0])  # alphabetically sort the data

    def convert(self, id_):
        with open(self.file, newline="") as f:
            reader = csv.reader(f)

            data = [
                row[:8] for row in reader
            ]  # the database contains some extra characteristics we don't need. Trim it to only the first 8 columns
            self.titles = data[0]  # the titles are in the first row
            data = data[1:][id_]

        return [(self.titles[i], data[i]) for i in self.relevant]

    def stringify(self, id_):
        return " ".join([str(i) for i in self.get_contest_by_id(id_)])

    def get_contest_by_name(self, name):
        """Retrieves the data for a contest by its name. If no contest is found returns None"""
        for contest in (
            self.data
        ):  # loop through the database until we find a contest with the right name
            if contest[0] == name:
                return contest

        return None

    def get_id(self, contest):
        return self.data.index(
            contest
        )  # ids are determined by the position of the contest in the list, so index the database to find the id

    def get_contest_by_id(self, id_):
        return self.data[
            id_
        ]  # to retrieve the contest using the id, just return the contest at position "id" of the list.

    def _search(self, string=""):
        """returns results from the database in terms of only how well the string matches up with it"""
        return list(
            reversed(
                sorted(
                    [contest for contest in self.data.copy()],
                    key=lambda x: similarity(string, x[0])
                    + 0.0001 * ord(x[0][0].lower()),
                )
            )
        )

    def get_date_of(self, id_):
        try:
            return find_dates.get_for_contest(self.get_contest_by_id(id_))

        except IndexError:
            pass

        except Exception as e:
            print(e, self.get_contest_by_id(id_))
            return None

        return "Not Found: " + str(self.get_contest_by_id(id_))

    def stringify_entry(self, id_):
        contest = self.get_contest_by_id(id_)
        return " ".join([str(i) for i in contest])

    def search(
        self,
        string="",
        grade_min=None,
        grade_max=None,
        subject=None,
        country=None,
        num_res=100,
    ):
        """Searches contests for "string" in the name, with a minimum grade matching "grade_min", a maximum grade matching "grade_max", a subject matching "subject"
        and a country matching "country". It returns "num_res" results (default 100)"""
        # we first use the search string to find a specific section of the database. We do this so that all the rest of the restrictions will take less time to run
        if string:
            filtered = self._search(string)

        else:  # this is only triggered if the string is empty or None.
            filtered = self.data.copy()

        # the following verifies that grade_min and grade_max work well with the actual maximum and minimum grade of all the contests.
        if grade_min and grade_max:
            grade_min = ContestList.GRADES[str(grade_min).lower()]
            grade_max = ContestList.GRADES[str(grade_max).lower()]
            if grade_min > grade_max:
                grade_min, grade_max = grade_max, grade_min

        if grade_min:
            f = []
            grade_min = ContestList.GRADES[str(grade_min).lower()]
            for row in filtered:
                try:
                    if (
                        row[6][1] >= grade_min
                    ):  # the maximum grade of the contest has to be greater than the minimum grade selected by the user (Otherwise the user might have selected (6, 7) with a contest that has grades (4, 5), and we might actually return the contest)
                        f.append(row)

                # if we encounter an error log it
                except Exception as e:
                    logging.error("Exception while verifying grade: %s", str(e))

            filtered = f

        if grade_max:
            f = []
            grade_max = ContestList.GRADES[str(grade_max).lower()]
            for row in filtered:
                try:
                    if (
                        row[6][0] <= grade_max
                    ):  # similarly to the last comment, the contest's minimum grade must be less than the user's choice of a maximum (Otherwise the user might have selected (0, 3) with a contest that has grades (4, 5), and we might actually return the contest)
                        f.append(row)

                # if we encounter an error log it
                except Exception as e:
                    logging.error("Exception while verifying grade: %s", str(e))

            filtered = f

        if subject:  # filter for subject by ensuring each contest has at least one subject which is in the user's requested list
            if isinstance(subject, str):
                subject = set((subject,))

            filtered = [row for row in filtered if row[8].intersection(subject)]

        if country:  # filter for country by ensuring each contest has at least one country which is in the user's requested list
            if isinstance(subject, str):
                subject = set((subject,))

            filtered = [row for row in filtered if row[1] in country]
        return filtered[:num_res]


class FreeResourceList:
    def __init__(self, file):
        self.file = file
        with open(file, encoding="utf-8", newline="\n") as f:
            reader = csv.reader(f)

            data = [
                row for row in reader
            ]  # the database contains some extra characteristics we don't need. Trim it to only the first 8 columns
            self.titles = data[0]  # the titles are in the first row
            data = data[1:]  # all the other rows are data

        new_data = []
        for entry in data:
            new = entry.copy()
            new[1] = new[1].split(", ")
            new[3] = set(new[3].split(", "))
            new[8] = new[8].replace(" years", "")

            if "-" in new[8]:
                new[8] = tuple([int(i.strip()) - 5 for i in new[8].split("-")])

            if "+" in new[8]:
                new[8] = (int(new[8].replace("+", "")) - 5, float("inf"))

            new[15] = "Project-Based" if new[15] == "Yes" else "Lesson-Based"
            new_data.append(new)

        self.relevant = 0, 1, 2, 4, 5, 6, 8, 9
        self.data = new_data

    def convert(self, id_):
        with open(self.file, encoding="utf-8", newline="\n") as f:
            reader = csv.reader(f)

            data = [
                row for row in reader
            ]  # the database contains some extra characteristics we don't need. Trim it to only the first 8 columns
            self.titles = data[0]  # the titles are in the first row
            data = data[1:][id_]

        return [(self.titles[i], data[i]) for i in self.relevant]

    def get_contest_by_name(self, name):
        """Retrieves the data for a contest by its name. If no contest is found returns None"""
        for contest in (
            self.data
        ):  # loop through the database until we find a contest with the right name
            if contest[0] == name:
                return contest

        return None

    def get_id(self, contest):
        return self.data.index(
            contest
        )  # ids are determined by the position of the contest in the list, so index the database to find the id

    def get_contest_by_id(self, id_):
        return self.data[
            id_
        ]  # to retrieve the contest using the id, just return the contest at position "id" of the list.

    def _search(self, string=""):
        """returns results from the database in terms of only how well the string matches up with it"""
        return list(
            reversed(
                sorted(
                    [contest for contest in self.data.copy()],
                    key=lambda x: similarity(string, x[0])
                    + 0.0001 * ord(x[0][0].lower()),
                )
            )
        )

    def stringify(self, id_):
        contest = self.get_contest_by_id(id_)
        a = [str(i) for i in contest]
        a[8] = str(a[8][0]) + "-" + str(a[8][1]) + " years"
        return " ".join(a)

    def search(
        self, string="", grade_min=None, grade_max=None, subject=None, num_res=100
    ):
        # we first use the search string to find a specific section of the database. We do this so that all the rest of the restrictions will take less time to run
        # grade min here is actually age min
        if string:
            filtered = self._search(string)

        else:  # this is only triggered if the string is empty or None.
            filtered = self.data.copy()

        if grade_min:
            f = []
            for row in filtered:
                try:
                    if (
                        row[8][1] >= grade_min
                    ):  # the maximum grade of the contest has to be greater than the minimum grade selected by the user (Otherwise the user might have selected (6, 7) with a contest that has grades (4, 5), and we might actually return the contest)
                        f.append(row)

                # if we encounter an error log it
                except Exception as e:
                    logging.error("Exception while verifying grade: %s", str(e))

            filtered = f

        if grade_max:
            f = []
            for row in filtered:
                try:
                    if (
                        row[8][0] <= grade_max
                    ):  # similarly to the last comment, the contest's minimum grade must be less than the user's choice of a maximum (Otherwise the user might have selected (0, 3) with a contest that has grades (4, 5), and we might actually return the contest)
                        f.append(row)

                # if we encounter an error log it
                except Exception as e:
                    logging.error("Exception while verifying grade: %s", str(e))

            filtered = f

        if subject:  # filter for subject by ensuring each contest has at least one subject which is in the user's requested list
            if isinstance(subject, str):
                subject = set((subject,))

            filtered = [row for row in filtered if row[3].intersection(subject)]

        return filtered[:num_res]


class AwardsList:  # Incomplete
    def __init__(self, file):
        self.file = file
        with open(file, encoding="utf-8", newline="\n") as f:
            reader = csv.reader(f)

            data = [
                row for row in reader
            ]  # the database contains some extra characteristics we don't need. Trim it to only the first 8 columns
            self.titles = data[0]  # the titles are in the first row
            data = data[1:]  # all the other rows are data

        new_data = []
        for entry in data:
            new = entry.copy()
            try:
                new[4] = float("".join([i for i in new[4] if i.isdigit() or i == "."]))

            except:
                pass

            new_data.append(new)

        self.data = new_data

        self.relevant = 0, 1, 2, 4, 7, 14, 16

    def convert(self, id_):
        with open(self.file, encoding="utf-8", newline="\n") as f:
            reader = csv.reader(f)

            data = [
                row for row in reader
            ]  # the database contains some extra characteristics we don't need. Trim it to only the first 8 columns
            self.titles = data[0]  # the titles are in the first row
            data = data[1:][id_]

        return [(self.titles[i], data[i]) for i in self.relevant]

    def get_contest_by_name(self, name):
        """Retrieves the data for a contest by its name. If no contest is found returns None"""
        for contest in (
            self.data
        ):  # loop through the database until we find a contest with the right name
            if contest[0] == name:
                return contest

        return None

    def get_id(self, contest):
        return self.data.index(
            contest
        )  # ids are determined by the position of the contest in the list, so index the database to find the id

    def get_contest_by_id(self, id_):
        return self.data[
            id_
        ]  # to retrieve the contest using the id, just return the contest at position "id" of the list.

    def _search(self, string=""):
        """returns results from the database in terms of only how well the string matches up with it"""
        return list(
            reversed(
                sorted(
                    [contest for contest in self.data.copy()],
                    key=lambda x: similarity(string, x[0])
                    + 0.0001 * ord(x[0][0].lower()),
                )
            )
        )

    def stringify(self, id_):
        contest = self.get_contest_by_id(id_)
        return " ".join([str(i) for i in contest])

    def search(self, string="", award_type=None, amount=None):
        # we first use the search string to find a specific section of the database. We do this so that all the rest of the restrictions will take less time to run
        if string:
            filtered = self._search(string)

        else:  # this is only triggered if the string is empty or None.
            filtered = self.data.copy()

        if award_type:  # filter for subject by ensuring each contest has at least one subject which is in the user's requested list
            if isinstance(subject, str):
                set((subject,))

            filtered = [row for row in filtered if row[2] in award_type]

        if amount:  # filter for subject by ensuring each contest has at least one subject which is in the user's requested list
            filtered = [
                row for row in filtered if isinstance(row[4], str) or row[4] >= amount
            ]

        return filtered[:num_res]


class CourseList:
    def __init__(self, filename):
        with open(file, encoding="utf-8", newline="\n") as f:
            reader = csv.reader(f)

            data = [
                row for row in reader
            ]  # the database contains some extra characteristics we don't need. Trim it to only the first 8 columns
            self.titles = data[0]  # the titles are in the first row
            data = data[1:]  # all the other rows are data


def checklinks(file, link_index):
    with open(file, newline="") as f:
        reader = csv.reader(f)

        data = [
            row for row in reader
        ]  # the database contains some extra characteristics we don't need. Trim it to only the first 8 columns
        data[0]  # the titles are in the first row
        data = data[1:]  # all the other rows are data

    for x, i in enumerate(data):
        url = i[link_index]
        try:
            response = requests.get(url)

        except Exception as e:
            print("Row:", x + 1)
            print(e)
            continue
        if response.status_code == 404:
            print("Row:", x + 1, response.status_code)


# checklinks(os.path.join('static', 'data', freeresources_filename), 22)
"""contests = ContestList(os.path.join('static', 'data', contest_filename))  # finish by creating an instance of ContestList
free_resources = FreeResourceList(os.path.join('static', 'data', freeresources_filename))
awards = AwardsList(os.path.join('static', 'data', awards_filename))
a = set()
for i in free_resources.data:
    a = a.union(i[3])

datas = [contests, free_resources, awards]
"""
import datetime


def excel_serial_to_date(serial):
    # Excel considers 1 as January 1, 1900, so we start from that date
    base_date = datetime.datetime(
        1899, 12, 30
    )  # Excel's date system starts from 1899-12-30
    return base_date + datetime.timedelta(days=int(serial))


"""
with open(os.path.join('static', 'data', broad_filename), newline = '') as f:
            reader = csv.reader(f)
            
            data = [row for row in reader]   # the database contains some extra characteristics we don't need. Trim it to only the first 8 columns 
            titles = data[0]   # the titles are in the first row
            data = data[1:]    # all the other rows are data

for i in data:
    if len(str(i[3])) == 5 and i[3].isdigit():
        a = excel_serial_to_date(i[3])
        i[3] = str(a.month) + '-' + str(a.day)

    if len(str(i[4])) == 5 and i[4].isdigit():
        a = excel_serial_to_date(i[4])
        i[4] = str(a.month) + '-' + str(a.day)

with open(os.path.join('static', 'data', broad_filename), newline = '\n', mode = 'w') as f:
            writer = csv.writer(f)
            
            writer.writerows([titles] + data)
"""


class BroadDatabase:
    def __init__(self, file):
        self.file = file
        with open(file, newline="\n") as f:
            reader = csv.reader(f)

            data = [
                row for row in reader
            ]  # the database contains some extra characteristics we don't need. Trim it to only the first 8 columns
            self.titles = data[0]  # the titles are in the first row
            self.titles[0] = self.titles[0][3:]
            self.data = data[1:]  # all the other rows are data

        for i in self.data:
            try:
                i[2] = i[2].split(", ")
                range_ = [
                    int(i)
                    for i in "".join(
                        [i for i in i[3] if i.isdigit() or i == "-"]
                    ).split("-")
                ]
                i[3] = "Ages " + str(list(range(range_[0], range_[1] + 1)))[1:-1]
                range_ = [
                    int(i) if i != "K" else "K"
                    for i in "".join(
                        [i for i in i[4] if i.isdigit() or i == "-" or i == "K"]
                    ).split("-")
                ]
                try:
                    i[4] = (
                        "Grades "
                        + str(
                            list(
                                range(
                                    range_[0] if isinstance(range_[0], int) else 0,
                                    range_[1] + 1,
                                )
                            )
                        )[1:-1]
                    )
                    i.append(
                        list(
                            range(
                                range_[0] if isinstance(range_[0], int) else 0,
                                range_[1] + 1,
                            )
                        )
                    )

                except:
                    i[4] = "Grades " + str(range_)[1:-1]
                    i.append(range_)

            except Exception as e:
                print("Deleting " + str(i) + ". Reason: " + str(e))
                del i

        self.words = []
        for i in self.data:
            for j in i:
                if isinstance(j, list):
                    if isinstance(j[0], int):
                        continue
                    self.words.extend([n.lower() for n in j])

                else:
                    for i in string.punctuation:
                        i = i.replace(i, " ")

                    while "  " in i:
                        i = i.replace("  ", " ")

                    i = i.lower()
                    self.words.extend(i.split(" "))

        self.relevant = 0, 1, 2, 3, 6, 8, 9
        self.types = str, str, list, str, str, str, str, str, str, str
        self.idfs = {i: self.words.count(i) for i in set(self.words)}
        self.titles.append("")

    def determine_tfidf(self, string_, id_or_contest):
        for i in string.punctuation:
            string_ = string_.replace(i, " ")

        while "  " in string_:
            string_ = string_.replace("  ", " ")

        string_ = [
            word.lower()
            for word in string_.split(" ")
            if word.lower() not in nltk.corpus.stopwords.words("english")
        ]
        # we have now processed the search query

        if isinstance(id_or_contest, int):
            test = [
                i.lower() if isinstance(i, str) else [j.lower() for j in i]
                for i in self.data[id_][:-1]
            ]

        else:
            test = [
                i.lower() if isinstance(i, str) else [j.lower() for j in i]
                for i in id_or_contest[:-1]
            ]

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
                    * max([i.count(word) for i in element])
                    / len("".join(element))
                    * len(word)
                    for element in test
                ]
            )

        return tfidf

    def _search(self, string="", subset=None):
        """returns results from the database in terms of only how well the string matches up with it"""

        if not subset:
            subset = self.data.copy()

        a = [
            self.determine_tfidf(string, x) + 0.0001 * ord(x[1][0].lower())
            for x in self.data
        ]
        search_results = [
            i[1]
            for i in sorted(list(zip(a, subset)), key=lambda x: x[0], reverse=True)
            if i[0] > 0.3 * len(string.split())
        ]
        return search_results

    def convert(self, result):
        result = result[:-1]
        r = [
            (self.titles[i], result[i])
            if self.types[i] == str
            else (self.titles[i], str(result[i])[1:-1].replace("'", ""))
            for i in self.relevant
        ]
        a = result[3].replace(",", "").split(" ")[1:]
        if a[0] != a[-1]:
            r[3] = self.titles[3], a[0] + "-" + a[-1]

        else:
            r[3] = self.titles[3], a[0]
        return r

    def search(self, string, num_res=100):
        # we first use the search string to find a specific section of the database. We do this so that all the rest of the restrictions will take less time to run
        filtered = self._search(string)
        return [self.convert(i) for i in filtered[:num_res]]

    def search_specific(self, query, ages, subjects_, types):
        a = set(range(ages[0], ages[1] + 1))
        print(a)
        asubs = set([i for i in subjects if subjects[i] in subjects_])

        filtered = [i for i in self.data if i[0] in types and asubs.intersection(i[2])]
        a = []
        for i in filtered:
            if min(i[-1]) > ages[1]:
                pass

            elif max(i[-1]) < ages[0]:
                pass

            else:
                a.append(i)

        filtered = a
        a = []
        for i in self._search(query, filtered):
            try:
                a.append(self.convert(i))
            except:
                pass

        return a


broad = BroadDatabase(os.path.join("static", "data", broad_filename))  # datas)

subjects = set()
for entry in broad.data:
    subjects = subjects.union(set(entry[2]))

subjects = [i for i in subjects if len(i.split(" ")) == 1]

subjects = {
    "Social Issues": "Social Studies",
    "Art": "Art",
    "Multimedia Reporting": "Media",
    "Environment": "Environmental Issues",
    "Sports": "Physical Education",
    "Nature": "Environmental Issues",
    "History": "Social Studies",
    "Creative Writing": "Writing",
    "Poetry": "Writing",
    "Filmmaking": "Media",
    "Creative Problem Solving": "Problem Solving",
    "Humor": "Writing",
    "English": "Language Arts",
    "Indigenous Youth": "Cultural Studies",
    "Community Service": "Community Engagement",
    "Literature": "Writing",
    "Storytelling": "Writing",
    "Personal Growth": "Personal Development",
    "Philanthropy": "Community Engagement",
    "Public Speaking": "Communication",
    "Journalism": "Media",
    "Social Studies": "Social Studies",
    "Physics": "Science",
    "Advanced Mathematics": "Math",
    "Programming": "Computer Science",
    "Arts": "Art",
    "STEM": "Science",
    "Math": "Math",
    "Community Engagement": "Community Engagement",
    "Leadership": "Leadership",
    "Essay Writing": "Writing",
    "Technology": "Technology",
    "Science": "Science",
    "Cultural": "Cultural Studies",
    "General": "General Studies",
    "General studies": "General Studies",
    "Global Issues": "Social Studies",
    "Computer Science": "Computer Science",
    "Game Design": "Art",
    "Engineering": "STEM",
    "Problem Solving": "Cognitive Skills",
    "Mathematics": "Math",
    "Environmental Issues": "Environmental Studies",
    "Writing": "Writing",
    "Environmental Literacy": "Environmental Studies",
    "Logic": "Math",
    "Photography": "Photography",
}


subject_list = set(subjects.values())

"""from pprint import pp

pp(broad.search_specific('', (1, 8), [], ['Competitions']))
"""
