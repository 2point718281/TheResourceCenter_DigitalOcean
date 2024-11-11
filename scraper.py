import requests
from bs4 import BeautifulSoup


def get_html(url, raw=False):
    """
    Fetch the html for a web page.

    :param url: URL of web page to get html of.
    :param raw: Whether to return the raw html or to wrap it in a BeautifulSoup object
    (default is False, so return value is by default a BeautifulSoup object)
    :return: A representation of the HTML of the page
    """
    html = requests.get(url).text
    if raw:
        return html
    return BeautifulSoup(html)
