import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useUnit } from '../../context/UnitContext';
import { Link } from 'react-router-dom';
import api from '../../services/api';
import StatCard from '../../components/StatCard';
import {
  Droplets,
  Sprout,
  Layers,
  CloudSun,
  MessageSquare,
  TrendingUp,
  AlertTriangle,
  ArrowRight,
  PlusCircle,
  Calendar
} from 'lucide-react';
import toast from 'react-hot-toast';

const FarmerDashboard = () => {
  const { user } = useAuth();
  const { areaUnit, formatVolume } = useUnit();
  const [farms, setFarms] = useState([]);
  const [history, setHistory] = useState([]);
  const [weather, setWeather] = useState(null);
  const [advisories, setAdvisories] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [farmsRes, historyRes, weatherRes, advRes] = await Promise.all([
        api.get('/farms').catch(() => ({ data: { farms: [] } })),
        api.get('/irrigation/history').catch(() => ({ data: { history: [] } })),
        api.get('/weather/current?lat=19.9975&lon=73.7898&location_name=Nashik').catch(() => ({ data: { weather: null } })),
        api.get('/advisories').catch(() => ({ data: { advisories: [] } }))
      ]);

      setFarms(farmsRes.data.farms || []);
      setHistory(historyRes.data.history || []);
      setWeather(weatherRes.data.weather || null);
      setAdvisories(advRes.data.advisories || []);
    } catch (err) {
      console.error(err);
      toast.error('Failed to load farmer dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const totalPlots = farms.reduce((sum, f) => sum + (f.plot_count || 0), 0);
  const latestPrediction = history[0];

  return (
    <div className="space-y-6">
      {/* Welcome Banner */}
      <div className="bg-gradient-to-r from-emerald-800 to-teal-700 rounded-2xl p-6 text-white shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <span className="text-emerald-200 text-xs font-semibold uppercase tracking-wider">
            Farmer Agro-Control Panel
          </span>
          <h1 className="text-2xl font-bold mt-1">
            Namaste, {user?.full_name || 'Kisan Mitra'}!
          </h1>
          <p className="text-xs text-emerald-100 mt-1 max-w-xl">
            Real-time irrigation schedule, weather telemetry, and specialist guidance for your farm plots.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <Link
            to="/farmer/irrigation"
            className="px-4 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-slate-950 rounded-xl text-xs font-bold shadow-md transition flex items-center space-x-1.5"
          >
            <Droplets className="w-4 h-4 text-slate-950" />
            <span>Run Prediction</span>
          </Link>
          <Link
            to="/farmer/farms"
            className="px-4 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl text-xs font-semibold border border-white/20 transition flex items-center space-x-1.5"
          >
            <PlusCircle className="w-4 h-4" />
            <span>Add Farm Plot</span>
          </Link>
        </div>
      </div>

      {/* Metric Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Registered Farms"
          value={farms.length}
          subtitle={`${totalPlots} active plots under management`}
          icon={Layers}
          color="emerald"
        />
        <StatCard
          title="Last Irrigation Need"
          value={latestPrediction ? `${latestPrediction.outputs?.net_irrigation_mm} mm` : '0 mm'}
          subtitle={latestPrediction ? formatVolume(latestPrediction.outputs?.net_irrigation_liters) : 'Run prediction below'}
          icon={Droplets}
          color="sky"
        />
        <StatCard
          title="Local Ambient Temp"
          value={weather ? `${weather.temperature_c}°C` : '28°C'}
          subtitle={weather ? `Humidity: ${weather.relative_humidity_pct}%` : 'Nashik Agro-Station'}
          icon={CloudSun}
          color="amber"
        />
        <StatCard
          title="Upcoming 3-Day Rain"
          value={weather ? `${weather.forecast_rainfall_3d_mm} mm` : '0.0 mm'}
          subtitle="Forecast offsets irrigation"
          icon={TrendingUp}
          color="indigo"
        />
      </div>

      {/* Live Weather & Quick Telemetry */}
      {weather && (
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-100 gap-2">
            <div>
              <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
                <CloudSun className="w-4 h-4 text-amber-500" />
                <span>Live Agro-Meteorological Station: {weather.location_name}</span>
              </h3>
              <p className="text-[11px] text-slate-500 mt-0.5">Source: {weather.source} • Lat: {weather.latitude}, Lon: {weather.longitude}</p>
            </div>
            <div className="text-xs bg-slate-100 px-2.5 py-1 rounded-md text-slate-700 font-medium">
              Reference ET0: <strong>{weather.et0_mm_day} mm/day</strong>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mt-4 text-center">
            {weather.daily_forecast?.map((day, idx) => (
              <div key={idx} className="bg-slate-50 p-3 rounded-xl border border-slate-100">
                <span className="text-[11px] font-bold text-slate-600 block">{day.date}</span>
                <span className="text-base font-extrabold text-slate-900 mt-1 block">
                  {day.max_temp}° / <span className="text-slate-500 text-xs">{day.min_temp}°</span>
                </span>
                <span className="text-[11px] text-sky-600 font-medium mt-1 flex items-center justify-center space-x-1">
                  <Droplets className="w-3 h-3" />
                  <span>{day.rain_mm} mm</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Two Column Layout: Advisories & Recent Predictions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Advisories & Alerts */}
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 text-emerald-600" />
              <span>Tailored Agricultural Advisories</span>
            </h3>
            <Link to="/farmer/advisories" className="text-xs text-emerald-600 font-semibold hover:underline">
              View All
            </Link>
          </div>

          <div className="space-y-3">
            {advisories.slice(0, 3).map((adv) => (
              <div
                key={adv.id}
                className={`p-3.5 rounded-xl border ${
                  adv.is_emergency ? 'bg-amber-50/70 border-amber-200' : 'bg-slate-50 border-slate-200'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-slate-900">{adv.title}</span>
                  {adv.is_emergency && (
                    <span className="bg-amber-200 text-amber-900 text-[9px] font-bold px-1.5 py-0.5 rounded uppercase">
                      Urgent
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed">{adv.advisory_text}</p>
                {adv.precautions && (
                  <p className="text-[11px] text-emerald-700 font-medium mt-1.5">
                    <strong>Precaution:</strong> {adv.precautions}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Recent Irrigation Calculations */}
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <Droplets className="w-4 h-4 text-sky-600" />
              <span>Recent Irrigation Predictions</span>
            </h3>
            <Link to="/farmer/irrigation" className="text-xs text-emerald-600 font-semibold hover:underline">
              New Calculation
            </Link>
          </div>

          {history.length === 0 ? (
            <div className="text-center py-10 bg-slate-50 rounded-xl border border-dashed border-slate-200">
              <Droplets className="w-8 h-8 text-slate-300 mx-auto mb-2" />
              <p className="text-xs text-slate-500">No predictions logged yet.</p>
              <Link
                to="/farmer/irrigation"
                className="mt-3 inline-block px-3 py-1.5 bg-emerald-600 text-white rounded-lg text-xs font-bold"
              >
                Calculate Irrigation for Plot
              </Link>
            </div>
          ) : (
            <div className="space-y-3">
              {history.slice(0, 3).map((item) => (
                <div key={item.id} className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 flex items-center justify-between">
                  <div>
                    <h5 className="text-xs font-bold text-slate-900">
                      {item.inputs?.crop_type} ({item.inputs?.crop_growth_stage})
                    </h5>
                    <p className="text-[11px] text-slate-500">
                      {item.inputs?.location_name} • {item.inputs?.farm_area} {item.inputs?.area_unit} ({item.inputs?.soil_type} soil)
                    </p>
                    <span className="text-[10px] text-slate-400">
                      Method: {item.outputs?.recommended_method} • Schedule: {item.outputs?.irrigation_frequency}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-sm font-extrabold text-emerald-600 block">
                      {item.outputs?.net_irrigation_mm} mm
                    </span>
                    <span className="text-[10px] text-slate-500 font-medium">
                      {formatVolume(item.outputs?.net_irrigation_liters)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FarmerDashboard;
