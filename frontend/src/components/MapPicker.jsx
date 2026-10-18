import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import { MapPin, Navigation } from 'lucide-react';

// Fix default leaflet marker icon issue in Webpack/Vite
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Component to handle map clicks
function LocationMarker({ position, setPosition, onSelect }) {
  useMapEvents({
    click(e) {
      const newPos = [e.latlng.lat, e.latlng.lng];
      setPosition(newPos);
      if (onSelect) {
        onSelect(e.latlng.lat, e.latlng.lng);
      }
    },
  });

  return position === null ? null : <Marker position={position}></Marker>;
}

const MapPicker = ({ initialLat = 19.9975, initialLon = 73.7898, onLocationSelected, height = "280px" }) => {
  const [position, setPosition] = useState([initialLat, initialLon]);

  useEffect(() => {
    if (initialLat && initialLon) {
      setPosition([initialLat, initialLon]);
    }
  }, [initialLat, initialLon]);

  const handleUseCurrentLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const lat = pos.coords.latitude;
          const lon = pos.coords.longitude;
          setPosition([lat, lon]);
          if (onLocationSelected) {
            onLocationSelected(lat, lon);
          }
        },
        (err) => {
          console.warn("Geolocation permission denied or unavailable:", err);
        }
      );
    }
  };

  return (
    <div className="relative rounded-xl overflow-hidden border border-slate-300 shadow-xs">
      <div style={{ height }}>
        <MapContainer
          center={position}
          zoom={11}
          scrollWheelZoom={false}
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <LocationMarker
            position={position}
            setPosition={setPosition}
            onSelect={onLocationSelected}
          />
        </MapContainer>
      </div>

      {/* Floating coordinates indicator & GPS button */}
      <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between pointer-events-none z-[1000]">
        <div className="bg-white/95 backdrop-blur-xs px-2.5 py-1 rounded-md shadow text-[11px] font-mono font-medium text-slate-700 pointer-events-auto border border-slate-200">
          <MapPin className="w-3 h-3 text-emerald-600 inline mr-1" />
          {position[0]?.toFixed(4)}, {position[1]?.toFixed(4)}
        </div>

        <button
          type="button"
          onClick={handleUseCurrentLocation}
          className="bg-emerald-600 hover:bg-emerald-700 text-white px-2.5 py-1 rounded-md shadow text-[11px] font-medium flex items-center space-x-1 pointer-events-auto transition"
          title="Detect GPS Location"
        >
          <Navigation className="w-3 h-3" />
          <span>GPS Pin</span>
        </button>
      </div>
    </div>
  );
};

export default MapPicker;
