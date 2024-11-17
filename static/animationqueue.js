let current_box = document.getElementById('box0')

let current_id = 1;

// Find all boxes that match
const regex = /^box\d+$/; // Example: IDs that start with 'myPrefix-' followed by digits

// Select all elements and filter by regex
const matchingElements = Array.from(document.querySelectorAll('div')).filter((el) => 
  regex.test(el.id)
);
const animation = 'fadein 0.1s forwards';

function nextbox() {
    matchingElements[current_id].style.animation = animation;
    current_box = matchingElements[current_id];
    current_id ++;
    current_box.addEventListener('animationend', nextbox)
}
current_box.style.animation = animation;
current_box.addEventListener('animationend', nextbox)