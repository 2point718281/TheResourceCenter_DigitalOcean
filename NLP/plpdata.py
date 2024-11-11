import docx
import re
import string

grade_p = re.compile("Grade \d")
data = docx.Document("PLPdata.docx")

total_text = "\n".join([i.text for i in data.paragraphs])

new = [i.strip() for i in re.split(grade_p, total_text)[1:]]

student_pattern = re.compile(
    "(?P<examplenum>\d+)"
    + re.escape(". ")
    + f"(?P<sname>[{string.ascii_lowercase} {string.ascii_uppercase}]+)"
    + re.escape(" – ")
    + f"[{string.ascii_lowercase} {string.ascii_uppercase}]+"
)


examples = []

for grade in new:
    examples.append(re.split(student_pattern, grade)[1:])

grades = []
for example in examples:
    parts = []
    for x, element in enumerate(example):
        if element.isdigit():
            parts.append(x)

    students = []
    for x, part in enumerate(parts):
        if x + 1 == len(parts):
            students.append(example[part:])

        else:
            students.append(example[part : parts[x + 1]])

    grades.append(students)
