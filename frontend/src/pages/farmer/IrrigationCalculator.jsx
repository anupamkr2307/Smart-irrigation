import React, { useState } from 'react';
import api from '../../services/api';
import { useUnit } from '../../context/UnitContext';
import MapPicker from '../../components/MapPicker';
import {
  Droplets,
  CloudSun,
  Search,
  Sparkles,
  Info,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Layers,
  Calendar,
  Compass,
  TrendingDown
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import toast from 'react-hot-toast';

const SOIL_OPTIONS = [
  { id: 'sandy', name: 'Sandy', desc: 'Fast drainage, low water holding (60 mm/m)' },
  { id: 'sandy loam', name: 'Sandy Loam', desc: 'Good drainage, moderate retention (100 mm/m)' },
  { id: 'loamy', name: 'Loamy (Ideal)', desc: 'Well-balanced fertility & water holding (140 mm/m)' },
  { id: 'silty', name: 'Silty', desc: 'Smooth, moderate infiltration, high fertility (160 mm/m)' },
  { id: 'clay loam', name: 'Clay Loam', desc: 'Dense, good water capacity, moderate drainage (160 mm/m)' },
  { id: 'clay', name: 'Clay (Black Soil)', desc: 'High retention, slow infiltration, waterlogging risk (180 mm/m)' },
];

const CROP_OPTIONS = [
  'Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane',
  'Tomato', 'Potato', 'Onion', 'Soybean', 'Chickpea', 'Mustard', 'Groundnut'
];

const STAGE_OPTIONS = ['Initial', 'Development', 'Mid-Season', 'Late Season'];

const IrrigationCalculator = () => {
  const { areaUnit, formatVolume } = useUnit();

  // Form State
  const [locationQuery, setLocationQuery] = useState('Nashik, Maharashtra');
  const [lat, setLat] = useState(19.9975);
  const [lon, setLon] = useState(73.7898);
  const [soilType, setSoilType] = useState('loamy');
  const [cropType, setCropType] = useState('Tomato');
  const [growthStage, setGrowthStage] = useState('Mid-Season');
  const [farmArea, setFarmArea] = useState(2.5);
  const [soilMoisture, setSoilMoisture] = useState(24.0);

  // Weather variables
  const [tempC, setTempC] = useState(27.5);
  const [humidity, setHumidity] = useState(48.0);
  const [recentRain, setRecentRain] = useState(0.0);
  const [forecastRain, setForecastRain] = useState(3.5);
  const [windSpeed, setWindSpeed] = useState(2.2);
  const [solarRad, setSolarRad] = useState(16.5);
  const [et0, setEt0] = useState(4.6);
  const [weatherSource, setWeatherSource] = useState('Default Baseline');

  // UI & Response state
  const [fetchingWeather, setFetchingWeather] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState(null);

  // Fetch live weather from Open-Meteo
  const handleFetchWeather = async () => {
    setFetchingWeather(true);
    try {
      // First geocode if text was changed
      let targetLat = lat;
      let targetLon = lon;
      let targetName = locationQuery;

      if (locationQuery) {
        const geoRes = await api.get(`/weather/geocode?query=${encodeURIComponent(locationQuery)}`);
        if (geoRes.data?.location) {
          targetLat = geoRes.data.location.latitude;
          targetLon = geoRes.data.location.longitude;
          targetName = `${geoRes.data.location.name}, ${geoRes.data.location.state || ''}`;
          setLat(targetLat);
          setLon(targetLon);
          setLocationQuery(targetName);
        }
      }

      const weatherRes = await api.get(`/weather/current?lat=${targetLat}&lon=${targetLon}&location_name=${encodeURIComponent(targetName)}`);
      if (weatherRes.data?.weather) {
        const w = weatherRes.data.weather;
        setTempC(w.temperature_c);
        setHumidity(w.relative_humidity_pct);
        setWindSpeed(w.wind_speed_ms);
        setRecentRain(w.recent_rainfall_mm);
        setForecastRain(w.forecast_rainfall_3d_mm);
        setEt0(w.et0_mm_day);
        setSolarRad(w.solar_radiation_mj);
        setWeatherSource(w.source);
        toast.success(`Weather updated for ${targetName} (${w.temperature_c}°C, ${w.relative_humidity_pct}% RH)`);
      }
    } catch (err) {
      console.error(err);
      toast.error('Failed to fetch weather. You may enter values manually.');
    } finally {
      setFetchingWeather(false);
    }
  };

  const handleMapLocationSelected = (selectedLat, selectedLon) => {
    setLat(selectedLat);
    setLon(selectedLon);
    toast.success(`Coordinates updated: ${selectedLat.toFixed(4)}, ${selectedLon.toFixed(4)}`);
  };

  // Run ML Prediction
  const handleCalculate = async (e) => {
    e.preventDefault();
    setPredicting(true);
    setResult(null);

    const payload = {
      location_name: locationQuery,
      soil_type: soilType,
      crop_type: cropType,
      crop_growth_stage: growthStage,
      farm_area: parseFloat(farmArea),
      area_unit: areaUnit,
      temperature_c: parseFloat(tempC),
      relative_humidity_pct: parseFloat(humidity),
      soil_moisture_pct: parseFloat(soilMoisture),
      recent_rainfall_mm: parseFloat(recentRain),
      forecast_rainfall_mm: parseFloat(forecastRain),
      wind_speed_ms: parseFloat(windSpeed),
      solar_radiation_mj: parseFloat(solarRad),
      evapotranspiration_et0_mm: parseFloat(et0)
    };

    try {
      const res = await api.post('/irrigation/predict', payload);
      setResult(res.data);
      toast.success('Irrigation prediction computed successfully!');
    } catch (err) {
      console.error(err);
      const msg = err.response?.data?.errors?.[0] || 'Prediction failed. Please check inputs.';
      toast.error(msg);
    } finally {
      setPredicting(false);
    }
  };

  // Prepare chart data
  const chartData = result ? [
    {
      name: 'Depth Comparison',
      'Crop Demand (ETc)': result.predictions.crop_evapotranspiration_etc_mm_day * 3,
      'Effective Rain': result.predictions.effective_rainfall_depth_mm,
      'Net Irrigation Needed': result.predictions.net_irrigation_depth_mm,
      'Gross Application': result.predictions.gross_irrigation_depth_mm
    }
  ] : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest flex items-center space-x-1">
            <Droplets className="w-3.5 h-3.5" />
            <span>FAO-56 Water Balance & Machine Learning Engine</span>
          </span>
          <h1 className="text-2xl font-extrabold text-slate-900 mt-1">Smart Irrigation Calculator</h1>
          <p className="text-xs text-slate-500 mt-1">
            Determine the exact water volume and schedule required for your crop to prevent water stress while maximizing water savings.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={handleFetchWeather}
            disabled={fetchingWeather}
            className="px-3.5 py-2 bg-sky-50 hover:bg-sky-100 text-sky-700 rounded-xl text-xs font-semibold border border-sky-200 transition flex items-center space-x-1.5 disabled:opacity-50"
          >
            <CloudSun className="w-4 h-4 text-sky-600" />
            <span>{fetchingWeather ? 'Fetching Live Weather...' : 'Auto-Fetch Weather'}</span>
          </button>
        </div>
      </div>

      {/* Main Grid: Inputs on Left, Interactive Map & Results on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Form Inputs (5 columns) */}
        <div className="lg:col-span-5 bg-white rounded-2xl p-6 border border-slate-200 shadow-xs space-y-5">
          <h3 className="text-sm font-bold text-slate-900 pb-2 border-b border-slate-100 flex items-center space-x-2">
            <Compass className="w-4 h-4 text-emerald-600" />
            <span>Farm & Environmental Parameters</span>
          </h3>

          <form onSubmit={handleCalculate} className="space-y-4">
            {/* Location with Geocode Button */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Farm Location / Village / City
              </label>
              <div className="flex items-center space-x-2">
                <input
                  type="text"
                  value={locationQuery}
                  onChange={(e) => setLocationQuery(e.target.value)}
                  placeholder="e.g. Nashik, Maharashtra"
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg outline-none focus:ring-2 focus:ring-emerald-500"
                />
                <button
                  type="button"
                  onClick={handleFetchWeather}
                  className="p-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs"
                  title="Search & Update Weather"
                >
                  <Search className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Crop & Growth Stage */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Crop Type</label>
                <select
                  value={cropType}
                  onChange={(e) => setCropType(e.target.value)}
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white"
                >
                  {CROP_OPTIONS.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Growth Stage</label>
                <select
                  value={growthStage}
                  onChange={(e) => setGrowthStage(e.target.value)}
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white"
                >
                  {STAGE_OPTIONS.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Farm Area */}
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs font-semibold text-slate-700">Farm Area</label>
                <span className="text-[10px] text-slate-500 font-semibold uppercase">Unit: {areaUnit}</span>
              </div>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="10000"
                value={farmArea}
                onChange={(e) => setFarmArea(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
              />
            </div>

            {/* Soil Type Selection */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Soil Texture Class</label>
              <select
                value={soilType}
                onChange={(e) => setSoilType(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white"
              >
                {SOIL_OPTIONS.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name} — {s.desc}
                  </option>
                ))}
              </select>
            </div>

            {/* Soil Moisture */}
            <div>
              <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                <span>Soil Moisture Level</span>
                <span className="text-emerald-700 font-bold">{soilMoisture}%</span>
              </div>
              <input
                type="range"
                min="5"
                max="45"
                step="1"
                value={soilMoisture}
                onChange={(e) => setSoilMoisture(e.target.value)}
                className="w-full accent-emerald-600"
              />
              <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                <span>Wilting Point (Dry)</span>
                <span>Optimum Field Capacity</span>
              </div>
            </div>

            {/* Weather snapshot fields (editable fallback) */}
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-[11px] font-bold text-slate-700 uppercase">Weather Snapshot</span>
                <span className="text-[10px] text-slate-500 font-mono">{weatherSource}</span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <label className="text-[10px] text-slate-500 block">Temperature (°C)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={tempC}
                    onChange={(e) => setTempC(e.target.value)}
                    className="w-full px-2 py-1 border border-slate-300 rounded bg-white"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-slate-500 block">Humidity (%)</label>
                  <input
                    type="number"
                    step="1"
                    value={humidity}
                    onChange={(e) => setHumidity(e.target.value)}
                    className="w-full px-2 py-1 border border-slate-300 rounded bg-white"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-slate-500 block">Forecast 3-Day Rain (mm)</label>
                  <input
                    type="number"
                    step="0.5"
                    value={forecastRain}
                    onChange={(e) => setForecastRain(e.target.value)}
                    className="w-full px-2 py-1 border border-slate-300 rounded bg-white"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-slate-500 block">Reference ET0 (mm/d)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={et0}
                    onChange={(e) => setEt0(e.target.value)}
                    className="w-full px-2 py-1 border border-slate-300 rounded bg-white"
                  />
                </div>
              </div>
            </div>

            <button
              type="submit"
              disabled={predicting}
              className="w-full py-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-sm font-bold shadow-md transition flex items-center justify-center space-x-2 disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4" />
              <span>{predicting ? 'Computing Agronomic Balance...' : 'Calculate Water Requirement'}</span>
            </button>
          </form>
        </div>

        {/* Right Column: Map & Results (7 columns) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Interactive Plot Map */}
          <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-xs">
            <div className="flex justify-between items-center mb-3">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center space-x-1.5">
                <Compass className="w-3.5 h-3.5 text-emerald-600" />
                <span>Farm Plot Geolocation (Click to Adjust Pin)</span>
              </h4>
              <span className="text-[11px] text-slate-500 font-mono">
                {lat.toFixed(4)}, {lon.toFixed(4)}
              </span>
            </div>
            <MapPicker
              initialLat={lat}
              initialLon={lon}
              onLocationSelected={handleMapLocationSelected}
              height="200px"
            />
          </div>

          {/* Results Display */}
          {result ? (
            <div className="bg-white rounded-2xl p-6 border border-emerald-200 shadow-md space-y-6 animate-fade-in">
              <div className="flex items-start justify-between pb-4 border-b border-slate-100">
                <div>
                  <span className="bg-emerald-100 text-emerald-800 text-[10px] font-bold px-2 py-0.5 rounded uppercase">
                    Prediction Generated
                  </span>
                  <h3 className="text-xl font-extrabold text-slate-900 mt-1">
                    {result.crop_type} ({result.crop_growth_stage})
                  </h3>
                  <p className="text-xs text-slate-500">
                    Area: {result.farm_area} {result.area_unit} ({result.area_sq_meters.toLocaleString()} m²) • Soil: {result.soil_type}
                  </p>
                </div>

                <div className="text-right">
                  <span className="text-2xl font-extrabold text-emerald-600">
                    {result.predictions.net_irrigation_depth_mm} mm
                  </span>
                  <span className="block text-xs font-bold text-slate-700">
                    {formatVolume(result.predictions.net_irrigation_volume_liters)}
                  </span>
                  <span className="text-[10px] text-slate-400">Net Water Volume</span>
                </div>
              </div>

              {/* Water Requirement Metrics Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                <div className="p-3 bg-emerald-50 rounded-xl border border-emerald-100">
                  <span className="text-[10px] font-bold text-emerald-700 uppercase">Gross Application</span>
                  <p className="text-lg font-bold text-emerald-900 mt-0.5">
                    {result.predictions.gross_irrigation_depth_mm} mm
                  </p>
                  <span className="text-[10px] text-emerald-600 font-medium">
                    {formatVolume(result.predictions.gross_irrigation_volume_liters)}
                  </span>
                </div>

                <div className="p-3 bg-sky-50 rounded-xl border border-sky-100">
                  <span className="text-[10px] font-bold text-sky-700 uppercase">Effective Rainfall</span>
                  <p className="text-lg font-bold text-sky-900 mt-0.5">
                    {result.predictions.effective_rainfall_depth_mm} mm
                  </p>
                  <span className="text-[10px] text-sky-600 font-medium">
                    {formatVolume(result.predictions.effective_rainfall_volume_liters)}
                  </span>
                </div>

                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                  <span className="text-[10px] font-bold text-slate-600 uppercase">Daily Crop ETc</span>
                  <p className="text-lg font-bold text-slate-900 mt-0.5">
                    {result.predictions.crop_evapotranspiration_etc_mm_day} mm/d
                  </p>
                  <span className="text-[10px] text-slate-500 font-medium">Evaporative loss</span>
                </div>

                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                  <span className="text-[10px] font-bold text-slate-600 uppercase">System Efficiency</span>
                  <p className="text-lg font-bold text-slate-900 mt-0.5">
                    {result.predictions.irrigation_efficiency_pct}%
                  </p>
                  <span className="text-[10px] text-slate-500 font-medium">{result.advisory.recommended_method}</span>
                </div>
              </div>

              {/* Water Balance Breakdown Chart */}
              <div>
                <h5 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2">
                  Agronomic Water Balance Depth (mm)
                </h5>
                <div className="h-48 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                      <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} unit=" mm" />
                      <Tooltip />
                      <Legend wrapperStyle={{ fontSize: '11px' }} />
                      <Bar dataKey="Crop Demand (ETc)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Effective Rain" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Net Irrigation Needed" fill="#10b981" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Gross Application" fill="#6366f1" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Schedule and Precautions */}
              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-2 text-xs">
                <div className="flex items-start space-x-2">
                  <Calendar className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <strong className="text-slate-900 block">Suggested Application Schedule:</strong>
                    <span className="text-slate-600">{result.advisory.suggested_schedule}</span>
                  </div>
                </div>

                <div className="flex items-start space-x-2 pt-2 border-t border-slate-200">
                  <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <strong className="text-slate-900 block">Soil-Specific Precaution ({result.soil_type}):</strong>
                    <span className="text-slate-600">{result.advisory.soil_precautions}</span>
                  </div>
                </div>
              </div>

              {/* ML Explainability & Model Confidence */}
              <div className="pt-2 border-t border-slate-100 text-[11px] text-slate-500 space-y-1">
                <p>
                  <strong>Model Confidence:</strong> {result.ml_diagnostics.model_version} • R²: {result.ml_diagnostics.r2_score} • Uncertainty: ±{result.ml_diagnostics.mae_uncertainty_mm} mm
                </p>
                <div>
                  <strong>Key Influencing Factors:</strong>
                  <ul className="list-disc list-inside mt-0.5 text-slate-600 space-y-0.5">
                    {result.ml_diagnostics.main_factors.map((f, i) => (
                      <li key={i}>{f}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-slate-50 rounded-2xl p-10 border border-dashed border-slate-200 text-center space-y-3">
              <Droplets className="w-10 h-10 text-emerald-600/40 mx-auto" />
              <h4 className="text-sm font-bold text-slate-700">Ready to Compute Water Requirements</h4>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                Select your crop, soil type, and farm area on the left. Click "Calculate Water Requirement" to run the ML regression pipeline.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default IrrigationCalculator;
