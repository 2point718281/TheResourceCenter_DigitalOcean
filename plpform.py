from plp import Questions, RADIO, TEXT, COMBO  # Imports

html = ""

for q in Questions:
    # handle each type of button differently
    if q["type"] == RADIO:
        name = q["id"]
        # template for HTML radio button
        addition = (
            '<input type="radio" id="{}" name="'
            + name
            + '" value={}><label for={}>{}</label><br>'
        )
        # add the options for the radio button
        for option in q["options"]:
            html += addition.format(option.lower(), option, option.lower(), option)

    elif q["type"] == TEXT:
        name = q["id"]
        # add the HTML for a text input and give it a unique id
        html += (
            f'<label for="{name}">'
            + q["desc"]
            + f'</label><br><input type="text" id="{name}" name="{name}"><br><br>'
        )

    elif q["type"] == COMBO:
        name = q["id"]

        # HTML start and end tags for the combobox (select) widget
        start = (
            f"<label for={name}>"
            + q["desc"]
            + f'</label><br><select name="{name}" id="{name}">'
        )
        end = "\n</select>"

        # Add the options for the combobox
        options = ""
        for option in q["options"]:
            options += f'\n\t<option value="{option.lower()}">{option}</option>'

        # Add all the HTML to the string
        html += start + options + end + "<br><br>"


# Submit button
html += '<input type="submit" value="Submit & Generate PLP">'
