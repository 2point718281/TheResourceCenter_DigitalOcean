class Section:
    def __init__(self, app, start_domain=""):
        self.domain = start_domain
        self.app = app

        self.sections = {}

    def add_section(self, section_name, section, methods=None):
        if methods:
            self.app.route("/" + self.domain + "/" + section_name, methods=methods)(
                section
            )

        else:
            self.app.route("/" + self.domain + "/" + section_name, methods=methods)(
                section
            )

        self.sections[section_name] = section

    def add(self, x, y, m=None):
        # alias for add_section
        self.add_section(x, y)
