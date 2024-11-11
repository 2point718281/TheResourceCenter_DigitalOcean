async function results()
{
    const subjects = ['math', 'science', 'coding', 'robotics', 'geography', 'writing', 'history', 'nature', 'film']
    const elements = [];
    subjects.forEach(function(element) {elements.push(document.getElementById(element).checked)});
    document.getElementById("textsearch").value
    const text = encodeURIComponent(document.getElementById("textsearch").value);
    const minage = encodeURIComponent(document.getElementById("minage").value);
    const maxage = encodeURIComponent(document.getElementById("maxage").value);
    let url = URL.concat('/results?q=', text, '&minage=', minage, '&maxage=', maxage);
    for (let i = 0; i < elements.length; i++) {
        if (elements[i]) {url = url.concat('&', subjects[i], '=', elements[i]);}
        
    }
    let results = await fetch(url).then(x => x.text());

    const results_elem = document.getElementById("results");
    results_elem.innerHTML = results

}