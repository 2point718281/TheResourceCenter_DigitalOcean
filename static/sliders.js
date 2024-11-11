function updateslider() {
    const minslider = document.getElementById("minage");
const maxslider = document.getElementById("maxage");
const minagel = document.getElementById("minagel");
const maxagel = document.getElementById("maxagel");

var minage = Number(minslider.value);
var maxage = Number(maxslider.value);


minagel.innerHTML = "Minimum Age: ".concat(minage);
maxagel.innerHTML = "Maximum Age: ".concat(maxage);}