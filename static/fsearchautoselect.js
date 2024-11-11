function set_box(box, value) {
    box.checked = value;
}

function update_children(parentcheckbox) {
    let checkboxid = parentcheckbox.id;
    console.log(checkboxid);
    let childcheckbox = document.getElementById(checkboxid + '0');
    console.log(childcheckbox);
    for (let i = 1; childcheckbox; i++) {
        console.log(childcheckbox);
        set_box(childcheckbox, parentcheckbox.checked);
        childcheckbox = document.getElementById(checkboxid + i.toString());
    }
}
function update_parent(childbox) {
    parent_box = document.getElementById(childbox.id.replace(/\d+$/, ''));
    children = [];
    let childcheckbox = document.getElementById(parent_box.id + '0');
    for (let i = 1; childcheckbox; i++) {
        children.push(childcheckbox);
        childcheckbox = document.getElementById(parent_box.id + i.toString());
    set_box(parent_box, children.every(element => element.checked));
    }
}