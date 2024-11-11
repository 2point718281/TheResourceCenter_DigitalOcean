let latitude, longitude; // Declare variables in the outer scope

// Check if geolocation is available
navigator.geolocation.getCurrentPosition(function(position) {
        // Assign latitude and longitude inside the callback
        latitude = position.coords.latitude;
        longitude = position.coords.longitude;

        // Output location
        console.log("Latitude: " + latitude);
        console.log("Longitude: " + longitude);
    }, function(error) {
        console.error("Error occurred: " + error.message);
    });

// You can access latitude and longitude here, but only after they are set


    const map = L.map('map').setView([latitude, longitude], 13);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);