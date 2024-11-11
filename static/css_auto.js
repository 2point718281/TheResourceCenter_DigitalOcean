var root = document.documentElement;

// SET AN EVENT LISTENER TO UPDATE CSS

document.addEventListener('resize', () => {
  root.style.setProperty('--screen-x', window.innerWidth)
  root.style.setProperty('--screen-y', window.innerHeight)
});
// DO IT ONCE NOW
root.style.setProperty('--screen-x', window.innerWidth);
root.style.setProperty('--screen-y', window.innerHeight);