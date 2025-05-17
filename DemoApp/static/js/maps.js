/**
 * JavaScript functions for working with Google Maps API
 */

let map;
let markers = [];
let infoWindow;

/**
 * Initialize Google Maps
 */
function initMap() {
  // Create a map centered on default location (will be updated with user location)
  map = new google.maps.Map(document.getElementById("map"), {
    center: { lat: 0, lng: 0 },
    zoom: 13,
    mapTypeControl: true,
    mapTypeControlOptions: {
      style: google.maps.MapTypeControlStyle.HORIZONTAL_BAR,
      position: google.maps.ControlPosition.TOP_RIGHT
    },
    fullscreenControl: true,
    streetViewControl: true,
    zoomControl: true
  });
  
  // Create a single reusable info window
  infoWindow = new google.maps.InfoWindow();
  
  // Try to get user's current location
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const userLocation = {
          lat: position.coords.latitude,
          lng: position.coords.longitude
        };
        
        // Center map on user location
        map.setCenter(userLocation);
        
        // Add marker for user location
        new google.maps.Marker({
          position: userLocation,
          map: map,
          icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale: 10,
            fillColor: "#4285F4",
            fillOpacity: 1,
            strokeColor: "#ffffff",
            strokeWeight: 2
          },
          title: "Your Location"
        });
      },
      () => {
        // Handle geolocation error
        console.log("Error: The Geolocation service failed.");
      }
    );
  }
}

/**
 * Find nearby clinics based on user location
 * @param {Object} location - The location to search around (lat/lng)
 */
function findNearbyClinics(location) {
  // Clear existing markers
  clearMarkers();
  
  // Create Places service
  const service = new google.maps.places.PlacesService(map);
  
  // Search for nearby clinics
  service.nearbySearch(
    {
      location: location,
      radius: 5000, // 5km radius
      type: ["hospital", "doctor", "health"]
    },
    (results, status) => {
      if (status === google.maps.places.PlacesServiceStatus.OK && results) {
        // Create markers for each result
        for (let i = 0; i < results.length; i++) {
          createClinicMarker(results[i], service);
        }
        
        // Display notification
        showNotification(`Found ${results.length} healthcare facilities nearby.`, "info");
      } else {
        showNotification("No healthcare facilities found nearby.", "warning");
      }
    }
  );
}

/**
 * Create a marker for a clinic
 * @param {Object} place - The place object from Google Places API
 * @param {Object} service - The Google Places service
 */
function createClinicMarker(place, service) {
  if (!place.geometry || !place.geometry.location) return;
  
  // Create marker
  const marker = new google.maps.Marker({
    map: map,
    position: place.geometry.location,
    title: place.name,
    animation: google.maps.Animation.DROP
  });
  
  // Add to markers array
  markers.push(marker);
  
  // Add click listener to show info window
  marker.addListener("click", () => {
    // Get place details
    service.getDetails(
      {
        placeId: place.place_id,
        fields: [
          "name",
          "formatted_address",
          "formatted_phone_number",
          "website",
          "opening_hours",
          "rating",
          "photos",
          "url"
        ]
      },
      (placeDetails, status) => {
        if (status === google.maps.places.PlacesServiceStatus.OK) {
          // Create info window content
          let content = `
            <div class="info-window">
              <h5>${placeDetails.name || place.name}</h5>
              <p><i class="fas fa-map-marker-alt"></i> ${placeDetails.formatted_address || "Address not available"}</p>
          `;
          
          if (placeDetails.formatted_phone_number) {
            content += `<p><i class="fas fa-phone"></i> ${placeDetails.formatted_phone_number}</p>`;
          }
          
          if (placeDetails.rating) {
            content += `<p><i class="fas fa-star"></i> ${placeDetails.rating} / 5</p>`;
          }
          
          // Add website link if available
          if (placeDetails.website) {
            content += `<p><a href="${placeDetails.website}" target="_blank" class="btn btn-sm btn-outline-primary">Visit Website</a></p>`;
          }
          
          // Add directions link
          content += `<p><a href="https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(
            placeDetails.formatted_address || place.name
          )}&destination_place_id=${place.place_id}" target="_blank" class="btn btn-sm btn-primary">Get Directions</a></p>`;
          
          content += `</div>`;
          
          // Set content and open info window
          infoWindow.setContent(content);
          infoWindow.open(map, marker);
        }
      }
    );
  });
}

/**
 * Clear all markers from the map
 */
function clearMarkers() {
  for (let i = 0; i < markers.length; i++) {
    markers[i].setMap(null);
  }
  markers = [];
}

/**
 * Search for an address and center the map on it
 * @param {string} address - The address to search for
 * @param {Function} callback - Optional callback function called with the found location
 */
function geocodeAddress(address, callback) {
  const geocoder = new google.maps.Geocoder();
  
  geocoder.geocode({ address: address }, (results, status) => {
    if (status === google.maps.GeocoderStatus.OK && results[0]) {
      const location = results[0].geometry.location;
      
      // Center map on the found location
      map.setCenter(location);
      
      // Add marker for the location
      const marker = new google.maps.Marker({
        map: map,
        position: location,
        animation: google.maps.Animation.DROP,
        title: address
      });
      
      // If callback is provided, call it with the found location
      if (typeof callback === "function") {
        callback(location, results[0]);
      }
    } else {
      showNotification(`Geocoding failed: ${status}`, "error");
    }
  });
}
