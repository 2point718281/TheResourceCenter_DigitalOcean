import subprocess
import sys


def main(module_name):
    # updating pip to latest version
    subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade pip"])

    # commanding terminal to pip install
    p = subprocess.call([sys.executable, "-m", "pip", "install", module_name])
    return p


try:
    pass

except:
    print("Installing required packages ('Flask')")
    main("Flask")

functionalities = {
    "search": True,
    "plp": False,
}  # list of functionalities and which to enable
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

import os
import logging
import base64

from user import users


if functionalities["plp"]:
    import plpform
    import plp

if functionalities["search"]:
    from data import broad, subject_list as subjects


app = Flask(__name__)
app.secret_key = base64.b64decode(os.environ.get("FLASK_SECRET"))
process = {}
dbs = {
    "Scholarship, Bursaries, and Awards": "/fsearch/sba",
    "Courses": "/fsearch/courses",
    "Competitions": "/fearch/competitions",
    "Free Educational Resources": "/fsearch/fer",
    "Extracurriculars, Clubs, and Volunteer": "/fsearch/extra-clubs-volun",
}


# subjects = list(subjects)

topicchooser = "\n".join(
    [
        f'<input type="checkbox" id="{topic}" name="{topic}" value="True"><label for="{topic}">{topic}</label><br>'
        for topic in subjects
    ]
)

logger = logging.getLogger(__file__)
# logging.basicConfig(filename='contestlist.log', encoding='utf-8', level=logging.DEBUG)


# NOTE THIS CODE REQUIRES Python 3.x
def load_template(file):
    filename = os.path.join("templates", file)
    with open(filename) as file:
        file = file.read()

    return file


app.jinja_env.globals.update(
    load=load_template,
    len=len,
    render=render_template,
    globals=globals,
    dbs=dbs,
    topics=topicchooser,
)


# Home page
def remove_punc(name):
    return "".join([i for i in name if i.isalpha() or i == " "])


@app.route("/")
def index():
    if session.get("LoggedIn", False):
        if session["perms"] == 3:
            return render_template("index.html").format(
                session["username"] + " (Admin)"
            )
        return render_template("index.html").format(session["username"])

    return render_template("index.html")


@app.route("/bsearch")
def broadsearch():
    if request.args.get("q", ""):
        return render_template(
            "bsearch.html",
            results=broad.search(request.args.get("q", "")),
            query=request.args.get("q", ""),
        )

    else:
        return render_template("bsearch.html", results=[], query="")


@app.route("/fsearch", methods=["GET", "POST"])
def focusedsearch():
    if request.method == "POST":
        query = request.form.get("q", "")
        rtype = request.form.get("rtype", list(dbs)[0])
        minage, maxage = (
            int(request.form.get("minage", 0)),
            int(request.args.get("maxage", 99)),
        )
        subjects_truefalse = {
            subject: (True if request.form.get(subject, "") else False)
            for subject in subjects
        }
        subjects_ = set(
            [subject for subject in subjects if subjects_truefalse[subject]]
        )
        types = {
            "Scholarship, Bursaries, and Awards": [
                "Scholarship",
                "Bursary",
                "Award",
                "Contest/Scholarship",
                "Scholarship/Bursary",
                "Competition/Scholarship",
                "Financial Aid",
            ],
            "Free Educational Resources": ["Free Educational Resource", "Courseware"],
            "Competitions": [
                "Contest",
                "Competition/Scholarship",
                "Contest/Scholarship",
            ],
            "Courses": ["Courseware"],
            "Extracurriculars, Clubs, and Volunteer": [],
        }

        if query or minage or maxage != 12 or subjects_:
            # try:
            return render_template(
                "fsearch.html",
                results=broad.search_specific(
                    query=query,
                    ages=(minage, maxage),
                    subjects_=subjects_,
                    types=types[rtype],
                ),
            )

            """except Exception as e 
                return str(e)
                return render_template('fsearch.html', results = [], query = '')"""

    else:
        return render_template("fsearch.html", results=[], query="")


@app.route("/plp", methods=["GET", "POST"])
def plp_():
    if request.method == "POST":
        print("hi", file=sys.stderr)
        session["plpname"] = request.form.get("name")
        filename = os.path.join(
            "plps", remove_punc(request.form.get("name")).replace(" ", "_") + "_PLP.pdf"
        )
        with open(os.path.join("static", "reloadplp.js")) as code:
            js = code.read()

        with open(os.path.join("static", "reloadplpc.js"), "w") as code:
            code.write(
                'const file = "'
                + "plp/view/"
                + remove_punc(request.form.get("name")).replace(" ", "_")
                + "_PLP.pdf"
                + '";'
                + js
            )

        # a = mp.Process(target=plp.generate_plp, args = (filename, request.form))
        # a.start()
        # process[filename] = a
        plp.generate_plp(filename, request.form)
        resp = make_response(render_template("view.html"), 200)
        resp.set_cookie("can_view", filename, httponly=True)
        return resp
    return render_template("plp.html", questions=plpform.html)


@app.route("/plp/view/<filename>/status")
def plpstatus(filename):
    if os.path.exists(
        os.path.join("plps", filename.replace("_PLP.pdf", "_status.txt"))
    ):
        if True:  # request.cookies.get('can_view') == filename:
            with open(
                os.path.join("plps", filename.replace("_PLP.pdf", "_status.txt"))
            ) as f:
                try:
                    if f.read().strip() == "done":
                        process[
                            os.path.join(
                                "plps", filename.replace("_PLP.pdf", "_status.txt")
                            )
                        ].join()
                        del process[
                            os.path.join(
                                "plps", filename.replace("_PLP.pdf", "_status.txt")
                            )
                        ]

                except:
                    pass
                return f.read().strip()

        else:
            return make_response("<h1>403 Access Forbidden</h1>", 403)

    else:
        return make_response("<h1>404 Not Found</h1>", 404)


@app.route("/plp/view/<filename>")
def viewplp(filename):
    if os.path.exists(os.path.join("plps", filename)):
        if True:  # request.cookies.get('can_view') == filename:
            return send_file(os.path.join("plps", filename), as_attachment=False)

        else:
            return make_response("<h1>403 Access Forbidden</h1>", 403)

    else:
        return make_response("<h1>404 Not Found</h1>", 404)


# ContestDatabase


@app.route("/contests")
def contests():
    if session.get("LoggedIn", False):
        if session["perms"] == 3:
            return render_template("searchsignedin.html").format(
                "<br>" + topicchooser, session["username"] + " (Admin)"
            )
        return render_template("searchsignedin.html").format(
            "<br>" + topicchooser, session["username"]
        )

    return render_template("search.html").format("<br>" + topicchooser)


@app.route("/contests/search")
def search():
    if request.args.get("contest", ""):
        titles = contests.titles[:-2]
        titles.append("Grades")
        information = contests.get_contest_by_id(int(request.args.get("contest")))[:-1]
        return "<br>".join(
            [
                f"<b>{titles[x]}:</b> {str(information[x])}<br>"
                for x in range(min(len(titles), len(information)))
            ]
        )
    all_ = request.args.to_dict()
    return redirect(url_for("results", **all_))


@app.route("/LogIn", methods=["GET", "POST"])
def LogIn():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["pwd"]
        login_successful = users.login(username, password)
        if login_successful.value != 0:
            session["LoggedIn"] = True
            session["username"] = username
            session["perms"] = login_successful.value
            return redirect(url_for("index"))

        else:
            return render_template("LogIn.html").format(
                'Login unsuccessful, check credentials or <a href = "/SignUp">Sign Up</a> with a new account'
            )

    return render_template("LogIn.html").format("")


@app.route("/LogOut")
def LogOut():
    del session["LoggedIn"]
    del session["username"]
    del session["perms"]
    return redirect(url_for("index"))


@app.route("/contests/results")
def results():
    query = request.args.get("q", "")
    minage, maxage = request.args.get("minage", 0), request.args.get("maxage", 99)
    subjects_truefalse = {
        subject: (True if request.args.get(subject, "") else False)
        for subject in subjects
    }
    subjects_ = set([subject for subject in subjects if subjects_truefalse[subject]])

    all_ = [
        "<b><a href="
        + url_for("search", contest=contests.get_id(i))
        + ">"
        + i[0]
        + "</a></b>"
        for i in contests.search(
            query, subject=subjects_, grade_min=minage, grade_max=maxage
        )
    ]
    logger.info(str(all_))
    return "<br>".join(
        all_
    )  # load_template('results.html').format(minage, maxage, topicchooser, '<br>'.join(all_))


if __name__ == "__main__":
    host, port = "0.0.0.0", 5000
    actual_host = host

    import webbrowser
    import socket

    if host == "0.0.0.0":
        actual_host = socket.gethostbyname(socket.gethostname())

    webbrowser.open(f"http://{actual_host}:{port}")
    app.run(host=host, port=port, debug=True)
