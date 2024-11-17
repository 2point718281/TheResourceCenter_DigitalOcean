# imports for string manipulation and searching for dates in strings
import re  # regex, for finding dates
import string  # for the constants built-in to the string module

import scraper  # for webscraping

import calendar  # for converting from the verbal form of dates to the numeric
import datetime  # for the datetime.datetime datastructure and because this module tells us what time it is

patterns = []  # a list of different patterns that dates can follow. Each element is a tuple, where the first element is a compiled pattern and the second is a function that converts the pattern to a datetime object

# constants
days = calendar.day_name
months = calendar.month_name[1:]
days_abbr = calendar.day_abbr
months_abbr = calendar.month_abbr[1:]

# This loop constructs the ordinal version of each date
dates = []

for place in range(1, 32):
    if str(place)[-1] == "1":
        dates.append(str(place) + "st")

    elif str(place)[-1] == "2":
        dates.append(str(place) + "nd")

    elif str(place)[-1] == "3":
        dates.append(str(place) + "rd")

    else:
        dates.append(str(place) + "th")

## EXCEPTIONS ##
dates[10] = "11th"
dates[11] = "12th"
dates[12] = "13th"


# These subpatterns are useful in multiple datepatterns
pattern_days = "(" + "|".join([re.escape(day.capitalize()) for day in days]) + ")"
pattern_months = (
    "(" + "|".join([re.escape(month.capitalize()) for month in months]) + ")"
)

pattern_days_abbr = (
    "(" + "|".join([re.escape(day.capitalize()) for day in days_abbr]) + ")"
)
pattern_months_abbr = (
    "(" + "|".join([re.escape(month.capitalize()) for month in months_abbr]) + ")"
)

pattern_dates = "(" + "|".join([re.escape(date.capitalize()) for date in dates]) + ")"

## Pattern 1: (Day of the week), (Month) (Date). This also accounts for any abbreviations. It assumes the year to be the current year

pattern_not_abbr = " ".join([pattern_days, pattern_months, pattern_dates])
pattern_day_abbr = " ".join([pattern_days_abbr, pattern_months, pattern_dates])
pattern_month_abbr = " ".join([pattern_days, pattern_months_abbr, pattern_dates])
pattern_both_abbr = " ".join([pattern_days_abbr, pattern_months_abbr, pattern_dates])


pattern = re.compile(
    "|".join(
        [pattern_not_abbr, pattern_day_abbr, pattern_month_abbr, pattern_both_abbr]
    )
)


# the function to convert matches of this pattern to a datetime object
def convert_match(match):
    """Converts "match" to a datetime"""

    for i in range(len(match) // 3):
        if match[i * 3]:
            match_type = i
            break
    values = match[
        match_type * 3 : (match_type + 1) * 3
    ]  # There are different types of matches that each have different parts abbreviated. We have to find out which one this is.

    # find the month
    months_ = months + months_abbr
    month = months_.index(values[1]) % 12 + 1

    # find the date
    day = int(
        values[2]
        .replace("th", "")
        .replace("st", "")
        .replace("nd", "")
        .replace("rd", "")
    )

    # this format doesn't tell us anything about the year so we just assume it to be the current year
    year = datetime.datetime.now().year

    return datetime.datetime(year, month, day)


patterns.append((pattern, convert_match))

## Pattern 2: (Month), (Date in numerical form), (Year in numerical form). This also accounts for month abbreviations.

pattern_month_abbr = pattern_months_abbr + " ([0-9]+) ([0-9][0-9][0-9][0-9])"
pattern_month = pattern_months + " ([0-9]+) ([0-9][0-9][0-9][0-9])"

pattern = re.compile("|".join([pattern_month_abbr, pattern_month]))


def convert_match(match):
    """Converts "match" to a datetime"""

    for i in range(len(match) // 3):
        if match[i * 3]:
            match_type = i
            break
    values = match[
        match_type * 3 : (match_type + 1) * 3
    ]  # There are different types of matches that each have different parts abbreviated. We have to find out which one this is.

    # find the month
    months_ = months + months_abbr
    month = months_.index(values[0]) % 12 + 1

    # find the date
    day = int(values[1])

    # find the year
    year = int(values[2])

    return datetime.datetime(year, month, day)


patterns.append((pattern, convert_match))

## Pattern 3: "available now"

# this is because sometimes the websites say a contest is "available now"
pattern = re.compile(re.escape(r"available now"))


def convert_match(match):
    # this match just returns the current date
    return datetime.datetime.now()


patterns.append((pattern, convert_match))

## Pattern 4: month, date

# this is like pattern 2 but without the year
pattern_month_abbr = pattern_months_abbr + " ([0-9]+)"
pattern_month = pattern_months + " ([0-9]+)"

# this is like pattern 2 but without the year and using ordinal numbers
pattern_month_abbr_dates = pattern_months_abbr + " " + pattern_dates
pattern_month_dates = pattern_months + " " + pattern_dates

pattern = re.compile(
    "|".join(
        [
            pattern_month_abbr,
            pattern_month,
            pattern_month_abbr_dates,
            pattern_month_dates,
        ]
    )
)


def convert_match(match):
    for i in range(len(match) // 2):
        if match[i * 2]:
            match_type = i
            break
    values = match[
        match_type * 2 : (match_type + 1) * 2
    ]  # This time each match type has two terms.

    # Find the month
    months_ = months + months_abbr
    month = months_.index(values[0]) % 12 + 1

    # Find the day
    day = int(
        values[1]
        .replace("st", "")
        .replace("th", "")
        .replace("nd", "")
        .replace("rd", "")
    )

    # We don't have information about the year so assume it's the current year
    year = datetime.datetime.now().year

    return datetime.datetime(year, month, day)


patterns.append((pattern, convert_match))


# The patterns we added don't account for puncuation. We need to remove puncuation.


def remove_punctuation(text):
    for char in string.punctuation:
        text = text.replace(char, "")

    return text


# This goes through each pattern and searches the string for it, returning any dates it finds. This function returns a generator
def findall_generator(text):
    text = remove_punctuation(text)

    for pattern in patterns:
        p, convert = pattern
        results = p.findall(text)
        for res in results:
            try:
                yield convert(res)

            except ValueError:
                continue


# This function is the same as the previous function but returns a list
def findall(text):
    return list(findall_generator(text))


# Finds the dates written on a contest's webpage
def get_for_contest(contest):
    html = scraper.get_html(contest[4], raw=True)
    return findall(html)
