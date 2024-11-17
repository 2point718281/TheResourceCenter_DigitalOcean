# Import necessary modules
import subprocess
import sys


def install_module(module_name):
    # Updating pip to latest version
    subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    # Installing required module
    p = subprocess.call([sys.executable, "-m", "pip", "install", module_name])
    return p


# Try importing Flask and install if not found
try:
    import flask
except ImportError:
    print("Installing required packages ('Flask')")
    install_module("Flask")

# Set functionalities to control the feature toggles
functionalities = {"search": True, "plp": True}
show_raw_search = True

import os
import logging  
import base64
import socket
import webbrowser
import env # Environment Variables
from NLP.preprocessing import get_filter

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template,
    session,
    make_response,
    send_file,
)
from waitress import serve

from user import users

# Conditional Imports
if functionalities["plp"]:
    import plpform
    import plp

if functionalities["search"]:
    from data import broad, broad_subject_mapping, rtypes

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename='TheResourceCenter.log', level=logging.INFO)
# loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# for logger in loggers:
#     logger.propagate = False
    
# Variables
dbs = {
    "Scholarship, Bursaries, and Awards": "/fsearch/sba",
    "Courses": "/fsearch/courses",
    "Competitions": "/fearch/competitions",
    "Free Educational Resources": "/fsearch/fer",
    "Extracurriculars, Clubs, and Volunteer": "/fsearch/extra-clubs-volun",
}
'''topicchooser = "\n".join(
    [
        f'<input type="checkbox" id="{topic}" name="{topic}" value="True"><label for="{topic}">{topic}</label><br>'
        for topic in subjects
    ]
)'''
topicchooser = ''


# Feature toggle settings
functionalities = {"search": True, "plp": False}

# Filter for autocorrect
autocorrect1, autocorrect2 = get_filter(False, False, False, True, False, True, False, True), get_filter(False, False, False, True, False, False, False, False)

# Helper function to load HTML templates
def load_template(file_name, mode = 'r'):
    """
    Load the HTML template from the templates directory.

    :param file_name: Name of the file to be loaded.

    :return: Contents of the HTML file.
    """
    file_path = os.path.join("templates", file_name)
    with open(file_path, mode) as file:
        read = file.read()
        if mode[-1] == 'b':
            read = read.decode()
        return read


# Create Flask app
app = Flask(__name__)
app.secret_key = base64.b64decode(
    os.environ.get("FLASK_SECRET", "")
)  # Ensure FLASK_SECRET is set
app.jinja_env.globals.update(
    load=load_template,
    len=len,
    str=str,
    render=render_template,
    globals=globals,
    dbs=dbs,
    topics=topicchooser,
    enumerate=enumerate,
)

# Database-related variables (mappings between topics and URLs)
dbs = {
    "Scholarship, Bursaries, and Awards": "/fsearch/sba",
    "Courses": "/fsearch/courses",
    "Competitions": "/fearch/competitions",
    "Free Educational Resources": "/fsearch/fer",
    "Extracurriculars, Clubs, and Volunteer": "/fsearch/extra-clubs-volun",
}

reviews = [
    "I think we have nothing like this",
    "It is very valuable resource for every student",
    "A single website with local information does not exist",
    "Kids/parents needs a trusted resource center",
    "I found the page and the information provided highly valuable and comprehensive.\nIt will be a valuable resource for the parents, students and teachers",
    "The Resource Centre will help any parent or even teachers who wants to support a child in achieving his best of the abilities",
    "I think that The Resource Center fulfills the need for students that want a personalized learning path.",
    "There is no other resource available like it.",
    "Covered all the areas which K-12 students really need",
    "It will help my son plan his course of action",
    "It is an absolutely great resource which provides a platform for students who want to participate in competitions and also provides comprehensive insight into personalized learning plan",
    "The Resource Center would be a useful to find resources to use in the classroom",
    "The curation of the resources has an education focus and would benefit students and educators",
]

reviewsleft = reviews[: len(reviews) // 2]
reviewsright = reviews[len(reviews) // 2 :]


class Section:
    """
    Class to manage different sections/routes in the Flask app.
    """

    def __init__(self, name, routes_dict, app):
        """
        Initialize a Section instance.

        :param name: Name of the section.
        :param routes_dict: Routes dictionary.
        :param app: Flask app instance.
        """
        for path in routes_dict:
            app.route(path, methods=routes_dict[path][2])(routes_dict[path][0])
        self.dict_ = routes_dict
        self.app = app
        self.name = name

    def add(self, path, func, desc, methods=["GET"]):
        """
        Add a route to the section.

        Args:
            path (str): Route path.
            func (function): Function to handle the route.
            desc (str): Description of the route.
            methods (list): HTTP methods allowed.
        """
        assert path not in self.dict_, "Path already exists in the section."
        self.dict_[path] = (func, desc, methods)
        self.app.route(path, methods=methods)(func)

    def __str__(self):
        """
        String representation of the Section instance.

        Returns:
            str: Section details.
        """
        return (
            self.name
            + "\n\t"
            + "\n\t".join([path + ": " + self.dict_[path][1] for path in self.dict_])
        )


class SectionManager:
    """
    Class to manage multiple sections within the Flask app.
    """

    def __init__(self, app):
        self.app = app
        self.sections = {}

    def add_section(self, name):
        """
        Add a new section to the manager.

        Args:
            name (str): Name of the section.

        Returns:
            Section: New Section instance.
        """
        new_section = Section(name, {}, self.app)
        self.sections[name] = new_section
        return new_section

    def __str__(self):
        """
        String representation of the SectionManager instance.

        Returns:
            str: Manager details.
        """
        manager_str = "Section Manager:"
        for section in self.sections.values():
            manager_str += "\n\t" + str(section).replace("\n", "\n\t")
        return manager_str


# Initialize SectionManager
section_manager = SectionManager(app)

# Authentication Section
auth_section = section_manager.add_section("Authentication")


def log_in():
    """
    Handle user login requests.

    :return: Flask response object.
    """
    return "This page has been disabled by a moderator"
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["pwd"]
        login_successful = users.login(username, password)
        if login_successful.value != 0:
            session["LoggedIn"] = True
            session["username"] = username
            session["perms"] = login_successful.value
            logger.info(f"User {username} logged in successfully.")
            return redirect(url_for("index"))
        logger.warning(f"Failed login attempt for user {username}.")
        return render_template("LogIn.html").format(
            'Login unsuccessful, check credentials or <a href="/SignUp">Sign Up</a> with a new account'
        )
    return render_template("LogIn.html").format("")


def log_out():
    """
    Handle user logout requests.

    :return: Flask response object.
    """
    return "This page has been disabled by a moderator"
    session.clear()
    logger.info("User logged out.")
    return redirect(url_for("index"))


# Add authentication routes
auth_section.add("/LogIn", log_in, "Log In route", ["GET", "POST"])
auth_section.add("/LogOut", log_out, "Log Out route")

# Home Section
home_section = section_manager.add_section("Home")


def index():
    """
    Render the home page.

    :return: Flask response object.
    """
    if session.get("LoggedIn", False):
        user_role = " (Admin)" if session["perms"] == 3 else ""
        return render_template("index.html").format(session["username"] + user_role)
    return render_template(
        "index.html", reviewsleft=reviewsleft, reviewsright=reviewsright
    )

def vision():
    return render_template('vision.html')

def founder():
    return render_template('founder.html')

def suggestions():
    """
    Render the suggestions page.

    :return: Flask response object.
    """
    return render_template("suggestions.html")

def ico():
    return send_file(os.path.join('static', 'theresourcecenterlogo.ico'))

# Add home route
home_section.add("/", index, "Home page route")
home_section.add("/founder", founder, "Founder's corner route")
home_section.add("/vision", vision, "Vision and Impact route")
home_section.add("/contact", suggestions, "Suggestion box route")
home_section.add("/favicon", ico, "Suggestion box route")

# Additional sections such as Search and PLP generation could be added here
# based on feature toggles (functionalities).
# Search Section
if functionalities["search"]:
    search_section = section_manager.add_section("Search")

    def broadsearch():
        """
        Renders the Broad Search page

        :return: Flask response object.
        """
        query = request.args.get("q", "")
        _, results = broad.search(query.lower().strip(), show_raw_search) if query else (query, [])
        new_query = autocorrect1(query)   # We've cached the results from the functions, so this won't take long
        query_ = autocorrect2(query)
        
        autocorrected = [(new_query[x] != query_[x], new_query[x]) for x in range(len(new_query))]
        if any([token[0] for token in autocorrected]):
            return render_template("bsearch.html", results=results, query=query, new_query=autocorrected)
        return render_template("bsearch.html", results=results, query=query)

    def focusedsearch():
        """
        Renders the Focused Search page

        :return: Flask response object.
        """
        if request.method == "POST":
            rtype = request.form.get("rtype", False)
            minage = int(request.form.get("minage", 0))
            maxage = int(request.form.get("maxage", 99))
            all_subjects = set(broad_subject_mapping.keys()).union(*[set(broad_subject_mapping[key]) for key in broad_subject_mapping])
            logger.info('ALL SUBJECTS: ', all_subjects)
            subjects_truefalse = {
                subject: bool(request.form.get(subject, False))
                for subject in all_subjects
            }
            subjects_ = set(
                [subject for subject in all_subjects if subjects_truefalse[subject]]
            )
            rtypes_ = set([rtype for rtype in rtypes if request.form.get(rtype, False)])
            ages = set([i for i in range(1, 13) if bool(request.form.get(str(i), False))])
            results = broad.search_specific(
                        query='',   # We are not using queries for focused search right now
                        ages=ages,
                        subjects_=subjects_,
                        rtypes=rtypes_,
                    )
            
            return render_template(
                "fsearch.html",
                results=results, broad_subjects=broad_subject_mapping, rtypes=rtypes, query="", grades = list(range(1, 13))
            )

        return render_template("fsearch.html", results=[], broad_subjects=broad_subject_mapping, rtypes=rtypes, query="", grades = list(range(1, 13)))


    # Add search routes to Search Section
    search_section.add("/bsearch", broadsearch, "Broad search route", ["GET", "POST"])
    search_section.add("/fsearch", focusedsearch, "Focused search route", ["GET", "POST"])

# PLP Section
plp_section = section_manager.add_section("PLP Generation")

def plp_():
    return render_template("plpcomingsoon.html")
    if request.method == "POST":
        session["plpname"] = request.form.get("name")
        filename = os.path.join(
            "plps", remove_punc(session["plpname"]).replace(" ", "_") + "_PLP.pdf"
        )
        with open(os.path.join("static", "reloadplp.js")) as code:
            js = code.read()

        with open(os.path.join("static", "reloadplpc.js"), "w") as code:
            code.write(
                'const file = "plp/view/'
                + remove_punc(session["plpname"]).replace(" ", "_")
                + '_PLP.pdf";'
                + js
            )

        plp.generate_plp(filename, request.form)
        resp = make_response(render_template("view.html"), 200)
        resp.set_cookie("can_view", filename, httponly=True)
        return resp
    return render_template("plp.html", questions=plpform.html)

def plpstatus(filename):
    status_file = os.path.join("plps", filename.replace("_PLP.pdf", "_status.txt"))
    if os.path.exists(status_file):
        with open(status_file) as f:
            return f.read().strip()
    return make_response("<h1>404 Not Found</h1>", 404)

def viewplp(filename):
    filepath = os.path.join("plps", filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=False)
    return make_response("<h1>404 Not Found</h1>", 404)

# Add PLP routes to PLP Section
plp_section.add("/plp", plp_, "PLP generation route", ["GET", "POST"])
plp_section.add("/plp/view/<filename>/status", plpstatus, "PLP status route")
plp_section.add("/plp/view/<filename>", viewplp, "View generated PLP route")


if __name__ == '__main__':
    print(section_manager)
    # Define host and port
    host, port = '0.0.0.0', 5000
    actual_host = socket.gethostbyname(socket.gethostname()) if host == '0.0.0.0' else host

    # Open default web browser
    logger.info(f"Opening web browser to http://{actual_host}:{port}")
    webbrowser.open(f'http://{actual_host}:{port}')

    # Run the Flask application
    app.run(host=host, port=port, debug=True)

'''
if __name__ == "__main__":
    host, port = "0.0.0.0", 80
    actual_host = (
        socket.gethostbyname(socket.gethostname()) if host == "0.0.0.0" else host
    )
    webbrowser.open(f"http://{actual_host}:{port}")
    serve(app, host=host, port=port)
'''
