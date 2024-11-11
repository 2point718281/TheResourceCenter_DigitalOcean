import unittest
import os
from collections import Counter

#### CONTEST DATABASE TESTS ####
import data


class DataTest(unittest.TestCase):
    def setUp(self):
        self.contests = data.ContestList("Contests_DataBase.csv")

    def test_ids(self):
        """
        Test that the ids are unique and are invertible
        """
        all_ = self.contests.data.copy()
        ids = []
        for name in all_:
            with self.subTest(contest=name[0]):
                id_ = self.contests.get_id(name)
                self.assertEqual(self.contests.get_contest_by_id(id_), name)
                ids.append(id_)

        c = list(Counter(ids).values())
        self.assertTrue(all([count == 1 for count in c]))

    def test_search_accuracy(self):
        """
        Test that if the name of a contest is searched directly, it will be the top result.
        """
        all_ = self.contests.data.copy()
        for name in all_:
            with self.subTest(contest=name[0]):
                self.assertEqual(self.contests.search(name[0])[0][0], name[0])

    def test_age_validation(self):
        """
        Tests to make sure that if the user enters invalid ages the search function will still work
        """
        self.contests.search(grade_min=12, grade_max="K")

    def test_age_search(self):
        """
        Tests that the search function takes into account the age selections
        """
        for min_age in range(0, 13):
            for max_age in range(min_age + 1, 13):
                grades = set(range(min_age, max_age + 1))
                with self.subTest(min_age=min_age, max_age=max_age):
                    results = self.contests.search(grade_min=min_age, grade_max=max_age)
                    for result in results:
                        with self.subTest(contest=result[0]):
                            contest_grades = set(range(result[6][0], result[6][1] + 1))
                            self.assertTrue(
                                bool(len(grades.intersection(contest_grades)))
                            )


#### USER DATABASE TESTS ####
import user


class UserTest(unittest.TestCase):
    def setUp(self):
        # Before every test we reset the user database
        self.users = user.UserDatabase("tests.db")
        self.users.add_account(
            user.date(1, 1, 1), "DummyForTesting", "D", "dummy@dummy.com"
        )
        self.users.add_account(
            user.date(1, 1, 1),
            "DummyAdmin",
            "admin",
            "admin@admin.com",
            user.Permissions.ADMIN,
        )

    def tearDown(self):
        # After every test we delete the user database because otherwise the userdatabase will recall from the previous session
        del (
            self.users
        )  # delete our handle on the object so we don't run into file errors
        os.remove("tests.db")  # delete the file.

    def test_account_addition(self):
        """
        Tests that accounts are properly added
        """
        accounts = len(self.users)  # the current number of accounts
        self.users.add_account(
            user.date(1, 1, 1), "DummyForTesting2", "D2", "dummy@gmail.com"
        )
        self.assertEqual(
            len(self.users.accounts), accounts + 1
        )  # after adding an account, we should have (previous number) + 1 accounts.

    def test_account_deletion(self):
        """
        Tests that accounts are properly deleted
        """
        accounts = len(self.users.accounts)  # the current number of accounts
        self.users.drop_user(None, "DummyForTesting")
        self.assertEqual(
            len(self.users.accounts), accounts - 1
        )  # after deleting an account, we should have (previous number) - 1 accounts.

    def test_login(self):
        """
        Tests that accounts are logged in with proper permissions
        """
        self.assertEqual(
            self.users.login("DummyAdmin", "admin"), user.Permissions.ADMIN
        )
        self.assertEqual(
            self.users.login("DummyForTesting", "D"), user.Permissions.USER
        )
        self.assertEqual(
            self.users.login("NonExistentAccount", ":]"), user.Permissions.LOGINFAILED
        )

    def test_last_id(self):
        """
        Tests that the get_last_id function is correctly implemented
        """
        self.assertEqual(self.users.get_last_id(), self.users.all_information[-1][0])

    def test_length(self):
        """
        Tests that len() returns the number of users in the database
        """
        self.assertEqual(len(self.users), len(self.users.accounts))


#### DATEFINDER TESTS ####
import find_dates


class DateFinderTest(unittest.TestCase):
    def setUp(self):
        self.contests = data.ContestList("Contests_DataBase.csv")
        self.results = []
        for contest_id in range(len(self.contests.data)):
            try:
                self.results.append(
                    find_dates.get_for_contest(
                        self.contests.get_contest_by_id(contest_id)
                    )
                )

            except:
                self.results.append([])
                print(self.contests.get_contest_by_id(contest_id))

    def test_is_datetime(self):
        """Tests that the find_dates module's get_for_contest function returns only datetime objects"""
        for contest_id in range(len(self.contests.data)):
            cond = all(
                [
                    isinstance(i, find_dates.datetime.datetime)
                    for i in self.results[contest_id]
                ]
            )
            self.assertTrue(cond)
            if not cond:
                print("Not a datetime, ", self.contests.get_contest_by_id(contest_id))

    def test_exists(self):
        """Tests that the find_dates module's get_for_contest function returns at least one result for each contest"""
        for x, results in enumerate(self.results):
            with self.subTest(contest=self.contests.get_contest_by_id(x)[0]):
                self.assertTrue(len(results) > 0)
                if len(results) == 0:
                    import webbrowser

                    webbrowser.open(self.contests.get_contest_by_id(x)[4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
