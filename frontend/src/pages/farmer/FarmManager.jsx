import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useUnit } from '../../context/UnitContext';
import MapPicker from '../../components/MapPicker';
import { Layers, Plus, MapPin, Trash2, Sprout, CheckCircle2, ChevronRight } from 'lucide-react';
import toast from 'react-hot-toast';

const FarmManager = () => {
  const { areaUnit } = useUnit();
  const [farms, setFarms] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddFarm, setShowAddFarm] = useState(false);
  const [selectedFarm, setSelectedFarm] = useState(null);
  const [showAddPlot, setShowAddPlot] = useState(false);

  // New farm form state
  const [farmName, setFarmName] = useState('');
  const [locationName, setLocationName] = useState('Nashik, Maharashtra');
  const [totalArea, setTotalArea] = useState(3.0);
  const [lat, setLat] = useState(19.9975);
  const [lon, setLon] = useState(73.7898);

  // New plot form state
  const [plotName, setPlotName] = useState('');
  const [plotArea, setPlotArea] = useState(1.5);
  const [plotSoil, setPlotSoil] = useState('loamy');
  const [plotCrop, setPlotCrop] = useState('Tomato');
  const [plotStage, setPlotStage] = useState('Mid-Season');
  const [plotMethod, setPlotMethod] = useState('Drip');

  useEffect(() => {
    loadFarms();
  }, []);

  const loadFarms = async () => {
    setLoading(true);
    try {
      const res = await api.get('/farms');
      setFarms(res.data.farms || []);
      if (res.data.farms?.length > 0 && !selectedFarm) {
        setSelectedFarm(res.data.farms[0]);
      }
    } catch (err) {
      console.error(err);
      toast.error('Failed to load farms');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateFarm = async (e) => {
    e.preventDefault();
    if (!farmName || !locationName) {
      toast.error('Farm name and location are required.');
      return;
    }

    try {
      const res = await api.post('/farms', {
        name: farmName,
        location_name: locationName,
        total_area: parseFloat(totalArea),
        area_unit: areaUnit,
        latitude: lat,
        longitude: lon,
      });
      toast.success('Farm registered successfully!');
      setShowAddFarm(false);
      setFarmName('');
      loadFarms();
    } catch (err) {
      console.error(err);
      toast.error('Failed to create farm');
    }
  };

  const handleAddPlot = async (e) => {
    e.preventDefault();
    if (!selectedFarm) return;
    if (!plotName) {
      toast.error('Plot name is required.');
      return;
    }

    try {
      await api.post(`/farms/${selectedFarm.id}/plots`, {
        plot_name: plotName,
        area: parseFloat(plotArea),
        soil_type: plotSoil,
        current_crop: plotCrop,
        growth_stage: plotStage,
        irrigation_system: plotMethod
      });
      toast.success('Plot added to farm!');
      setShowAddPlot(false);
      setPlotName('');
      loadFarms();
    } catch (err) {
      console.error(err);
      toast.error('Failed to add plot');
    }
  };

  const handleDeleteFarm = async (farmId) => {
    if (!window.confirm('Are you sure you want to delete this farm and all its plots?')) return;
    try {
      await api.delete(`/farms/${farmId}`);
      toast.success('Farm removed');
      loadFarms();
    } catch (err) {
      toast.error('Failed to delete farm');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900">Farm & Plot Management</h1>
          <p className="text-xs text-slate-500 mt-1">Configure your agricultural land parcels, soil attributes, and crop varieties.</p>
        </div>

        <button
          onClick={() => setShowAddFarm(!showAddFarm)}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold shadow-xs transition flex items-center space-x-1.5"
        >
          <Plus className="w-4 h-4" />
          <span>{showAddFarm ? 'Cancel' : 'Register New Farm'}</span>
        </button>
      </div>

      {/* New Farm Drawer/Modal */}
      {showAddFarm && (
        <div className="bg-white rounded-2xl p-6 border border-emerald-200 shadow-sm animate-fade-in">
          <h3 className="text-sm font-bold text-slate-900 mb-4 pb-2 border-b border-slate-100">
            Register a New Farm Parcel
          </h3>
          <form onSubmit={handleCreateFarm} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Farm Name *</label>
                <input
                  type="text"
                  required
                  value={farmName}
                  onChange={(e) => setFarmName(e.target.value)}
                  placeholder="e.g. Riverbank Orchard"
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Location Name *</label>
                <input
                  type="text"
                  required
                  value={locationName}
                  onChange={(e) => setLocationName(e.target.value)}
                  placeholder="e.g. Nashik, Maharashtra"
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Total Area ({areaUnit})</label>
                <input
                  type="number"
                  step="0.1"
                  required
                  value={totalArea}
                  onChange={(e) => setTotalArea(e.target.value)}
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Pin Location on Map ({lat.toFixed(4)}, {lon.toFixed(4)})
              </label>
              <MapPicker
                initialLat={lat}
                initialLon={lon}
                onLocationSelected={(newLat, newLon) => {
                  setLat(newLat);
                  setLon(newLon);
                }}
                height="180px"
              />
            </div>

            <div className="flex justify-end space-x-2 pt-2">
              <button
                type="button"
                onClick={() => setShowAddFarm(false)}
                className="px-4 py-2 border border-slate-300 text-slate-600 rounded-lg text-xs font-semibold"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-xs font-bold"
              >
                Save Farm
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Main Farm List & Plot Explorer */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Farm List */}
        <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-xs space-y-3">
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2">My Farms</h3>
          {farms.length === 0 ? (
            <p className="text-xs text-slate-500 py-4 text-center">No farms registered yet.</p>
          ) : (
            farms.map((farm) => {
              const isSelected = selectedFarm?.id === farm.id;
              return (
                <div
                  key={farm.id}
                  onClick={() => setSelectedFarm(farm)}
                  className={`p-3.5 rounded-xl border cursor-pointer transition ${
                    isSelected
                      ? 'bg-emerald-50/70 border-emerald-300 shadow-xs'
                      : 'bg-slate-50 border-slate-200 hover:border-slate-300'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <h4 className="text-xs font-bold text-slate-900">{farm.name}</h4>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteFarm(farm.id);
                      }}
                      className="text-slate-400 hover:text-red-600 p-1"
                      title="Delete Farm"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <p className="text-[11px] text-slate-500 mt-1 flex items-center space-x-1">
                    <MapPin className="w-3 h-3 text-slate-400" />
                    <span>{farm.location_name}</span>
                  </p>
                  <div className="flex justify-between text-[11px] text-slate-600 mt-2 pt-2 border-t border-slate-200/60">
                    <span>{farm.total_area} {farm.area_unit}</span>
                    <span className="font-semibold text-emerald-700">{farm.plots?.length || 0} Plots</span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Right: Plots of Selected Farm */}
        <div className="lg:col-span-2 bg-white rounded-2xl p-6 border border-slate-200 shadow-xs space-y-4">
          {selectedFarm ? (
            <>
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center pb-4 border-b border-slate-100 gap-2">
                <div>
                  <h3 className="text-lg font-bold text-slate-900">{selectedFarm.name}</h3>
                  <p className="text-xs text-slate-500">
                    Location: {selectedFarm.location_name} • Coordinates: {selectedFarm.latitude?.toFixed(4)}, {selectedFarm.longitude?.toFixed(4)}
                  </p>
                </div>
                <button
                  onClick={() => setShowAddPlot(!showAddPlot)}
                  className="px-3 py-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-800 rounded-lg text-xs font-semibold border border-emerald-200 transition flex items-center space-x-1"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>{showAddPlot ? 'Cancel' : 'Add Plot'}</span>
                </button>
              </div>

              {/* Add Plot Form */}
              {showAddPlot && (
                <form onSubmit={handleAddPlot} className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-3 animate-fade-in">
                  <h5 className="text-xs font-bold text-slate-800 uppercase tracking-wider">New Plot Details</h5>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-700 mb-1">Plot Name</label>
                      <input
                        type="text"
                        required
                        value={plotName}
                        onChange={(e) => setPlotName(e.target.value)}
                        placeholder="Plot B - South Field"
                        className="w-full px-2.5 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                      />
                    </div>
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-700 mb-1">Plot Area ({areaUnit})</label>
                      <input
                        type="number"
                        step="0.1"
                        required
                        value={plotArea}
                        onChange={(e) => setPlotArea(e.target.value)}
                        className="w-full px-2.5 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                      />
                    </div>
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-700 mb-1">Soil Texture</label>
                      <select
                        value={plotSoil}
                        onChange={(e) => setPlotSoil(e.target.value)}
                        className="w-full px-2.5 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                      >
                        <option value="sandy">Sandy</option>
                        <option value="sandy loam">Sandy Loam</option>
                        <option value="loamy">Loamy</option>
                        <option value="silty">Silty</option>
                        <option value="clay loam">Clay Loam</option>
                        <option value="clay">Clay</option>
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-700 mb-1">Current Crop</label>
                      <input
                        type="text"
                        value={plotCrop}
                        onChange={(e) => setPlotCrop(e.target.value)}
                        placeholder="Wheat / Tomato / Cotton"
                        className="w-full px-2.5 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                      />
                    </div>
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-700 mb-1">Growth Stage</label>
                      <select
                        value={plotStage}
                        onChange={(e) => setPlotStage(e.target.value)}
                        className="w-full px-2.5 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                      >
                        <option value="Initial">Initial</option>
                        <option value="Development">Development</option>
                        <option value="Mid-Season">Mid-Season</option>
                        <option value="Late Season">Late Season</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-700 mb-1">Irrigation System</label>
                      <select
                        value={plotMethod}
                        onChange={(e) => setPlotMethod(e.target.value)}
                        className="w-full px-2.5 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                      >
                        <option value="Drip">Drip Irrigation (90% eff)</option>
                        <option value="Sprinkler">Micro-Sprinkler (75% eff)</option>
                        <option value="Furrow">Furrow Irrigation (60% eff)</option>
                        <option value="Basin">Basin / Controlled (50% eff)</option>
                      </select>
                    </div>
                  </div>

                  <div className="flex justify-end space-x-2 pt-2">
                    <button
                      type="submit"
                      className="px-3 py-1.5 bg-emerald-600 text-white rounded-lg text-xs font-bold"
                    >
                      Save Plot
                    </button>
                  </div>
                </form>
              )}

              {/* List of Plots in selected farm */}
              <div className="space-y-3">
                {selectedFarm.plots?.length === 0 ? (
                  <p className="text-xs text-slate-500 py-6 text-center">No plots added to this farm yet.</p>
                ) : (
                  selectedFarm.plots?.map((plot) => (
                    <div key={plot.id} className="p-4 bg-slate-50 rounded-xl border border-slate-200 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                      <div>
                        <div className="flex items-center space-x-2">
                          <h4 className="text-xs font-bold text-slate-900">{plot.plot_name}</h4>
                          <span className="bg-emerald-100 text-emerald-800 text-[10px] font-semibold px-2 py-0.5 rounded">
                            {plot.current_crop || 'Unplanted'} ({plot.growth_stage})
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-500 mt-1">
                          Area: {plot.area} {selectedFarm.area_unit} • Soil: {plot.soil_type} (pH: {plot.soil_ph})
                        </p>
                      </div>

                      <div className="text-right">
                        <span className="text-xs font-bold text-slate-700 block">
                          Method: {plot.irrigation_system}
                        </span>
                        <span className="text-[10px] text-slate-500">
                          Target Moisture: {plot.target_moisture_pct}%
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </>
          ) : (
            <p className="text-xs text-slate-500 text-center py-10">Select a farm from the left to view its plots.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default FarmManager;
