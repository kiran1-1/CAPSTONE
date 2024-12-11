// to get current year
function getYear() {
    var currentDate = new Date();
    var currentYear = currentDate.getFullYear();
    document.querySelector("#displayYear").innerHTML = currentYear;
}

getYear();


//  owl carousel script
$(".owl-carousel").owlCarousel({
    loop: true,
    margin: 20,
    nav: true,
    navText: [],
    autoplay: true,
    autoplayHoverPause: true,
    responsive: {
        0: {
            items: 1
        },
        1000: {
            items: 2
        }
    }
});

//    end owl carousel script 



document.addEventListener("DOMContentLoaded", function () {
    const dropdown = document.getElementById("dropdown-menu");
    const parkingInfo = document.getElementById("parking-info");
  
    dropdown.addEventListener("change", function () {
      const lot = dropdown.value;
      if (!lot) return;
  
      // Send a request to start processing the selected lot
      fetch('/start_processing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lot: lot }),
      })
      .then(response => response.json())
      .then(data => {
        console.log(data.message);
        fetchLiveData(lot); // Start fetching live data
      })
      .catch(error => console.error('Error:', error));
    });
  
    function fetchLiveData(lot) {
      setInterval(() => {
        fetch(`/get_live_data?lot=${lot}`)
          .then(response => response.json())
          .then(data => {
            console.log("Live Data Received:", data); // Debugging response
            document.getElementById("parking-info").classList.remove("hidden"); 
            // Update the parking slot counts dynamically
            animateCounter("available-slots", data.available);
            animateCounter("occupied-slots", data.occupied);
            animateCounter("disabled-slots", data.disabled);
          })
          .catch(error => console.error('Error fetching live data:', error));
      }, 2000); // Fetch updates every 2 seconds
    }
  
    function animateCounter(elementId, endValue) {
      const element = document.getElementById(elementId);
      let currentValue = parseInt(element.textContent) || 0;
      const increment = Math.ceil((endValue - currentValue) / 20);
  
      const interval = setInterval(() => {
        currentValue += increment;
        if (currentValue >= endValue) {
          currentValue = endValue;
          clearInterval(interval);
        }
        element.textContent = currentValue;
      }, 50); // Update interval in milliseconds
    }
  });




/** google_map js **/
function myMap() {
    var mapProp = {
        center: new google.maps.LatLng(40.712775, -74.005973),
        zoom: 18,
    };
    var map = new google.maps.Map(document.getElementById("googleMap"), mapProp);
}