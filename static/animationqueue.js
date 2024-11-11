let current_box = document.getElementById('box0')

let current_id = 1;

// Find all boxes that match
const regex = /^box\d+$/; // Example: IDs that start with 'myPrefix-' followed by digits

// Select all elements and filter by regex
const matchingElements = Array.from(document.querySelectorAll('div')).filter((el) => 
  regex.test(el.id)
);

function nextbox() {
    matchingElements[current_id].style.animation = 'fade 0.1s forwards';
    current_box = matchingElements[current_id];
    current_id ++;
    current_box.addEventListener('animationend', nextbox)
}

current_box.addEventListener('animationend', nextbox)