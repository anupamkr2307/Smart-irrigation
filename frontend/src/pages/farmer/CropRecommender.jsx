import React, { useState } from 'react';
import api from '../../services/api';
import { Sprout, Award, Droplets, Thermometer, ShieldAlert, CheckCircle, Info, Sparkles } from 'lucide-react';
import toast from 'react-hot-toast';

const CropRecommender = () => {
  const [soilType, setSoilType] = useState('loamy');
  const [soilPh, setSoilPh] = useState(6.8);
  const [temperature, setTemperature] = useState(26.0);
  const [humidity, setHumidity] = useState(50.0);
  const [waterAvailability, setWaterAvailability] = useState('Medium');

  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState(null);

  const handleRecommend = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await api.post('/crops/recommend', {
        soil_type: soilType,
        soil_ph: parseFloat(soilPh),
        temperature_c: parseFloat(temperature),
        relative_humidity_pct: parseFloat(humidity),
        water_availability: waterAvailability
      });
      setRecommendations(res.data.recommendations || []);
      toast.success('Crop suitability evaluated successfully!');
    } catch (err) {
      console.error(err);
      toast.error('Failed to get crop recommendations');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs">
        <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest flex items-center space-x-1">
          <Sprout className="w-3.5 h-3.5" />
          <span>Multi-Criteria Agronomic Suitability Engine</span>
        </span>
        <h1 className="text-2xl font-extrabold text-slate-900 mt-1">Crop Recommendation Engine</h1>
        <p className="text-xs text-slate-500 mt-1">
          Identify the top 5 most viable and profitable crops for your specific soil type, pH level, temperature range, and water availability.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Input Parameters (4 cols) */}
        <div className="lg:col-span-4 bg-white rounded-2xl p-6 border border-slate-200 shadow-xs space-y-4">
          <h3 className="text-sm font-bold text-slate-900 pb-2 border-b border-slate-100">
            Agronomic Field Conditions
          </h3>

          <form onSubmit={handleRecommend} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Soil Texture Class</label>
              <select
                value={soilType}
                onChange={(e) => setSoilType(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white"
              >
                <option value="sandy">Sandy (Light, High Drainage)</option>
                <option value="sandy loam">Sandy Loam (Good Drainage)</option>
                <option value="loamy">Loamy (Balanced & Fertile)</option>
                <option value="silty">Silty (Fine Texture)</option>
                <option value="clay loam">Clay Loam (Moderate Retention)</option>
                <option value="clay">Clay (Heavy Black Soil)</option>
              </select>
            </div>

            <div>
              <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1">
                <span>Soil pH Level</span>
                <span className="text-emerald-700 font-bold">{soilPh}</span>
              </div>
              <input
                type="range"
                min="4.5"
                max="9.0"
                step="0.1"
                value={soilPh}
                onChange={(e) => setSoilPh(e.target.value)}
                className="w-full accent-emerald-600"
              />
              <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                <span>Acidic (&lt;6.0)</span>
                <span>Neutral (6.5-7.5)</span>
                <span>Alkaline (&gt;8.0)</span>
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Expected Seasonal Temp (°C)</label>
              <input
                type="number"
                step="0.5"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Water Supply & Availability</label>
              <select
                value={waterAvailability}
                onChange={(e) => setWaterAvailability(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white"
              >
                <option value="High">High (Borewell / Canal / Ample Storage)</option>
                <option value="Medium">Medium (Seasonal Well / Regulated Drip)</option>
                <option value="Low">Low (Rainfed / Severe Scarcity)</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold shadow-xs transition flex items-center justify-center space-x-1.5 disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4" />
              <span>{loading ? 'Evaluating Suitability...' : 'Analyze & Rank Crops'}</span>
            </button>
          </form>

          <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-[11px] text-slate-500 space-y-1">
            <span className="font-bold text-slate-700 block flex items-center space-x-1">
              <Info className="w-3.5 h-3.5 text-emerald-600" />
              <span>Multi-Factor Scoring Matrix</span>
            </span>
            <p>Scores reflect agronomic compatibility with soil texture (35%), pH (20%), temperature window (25%), and water supply feasibility (20%).</p>
          </div>
        </div>

        {/* Results: Top 5 Suitable Crops (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          {recommendations ? (
            <div className="space-y-4">
              <div className="flex justify-between items-center px-1">
                <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
                  Top 5 Recommended Crops
                </h3>
                <span className="text-xs text-slate-500 font-medium">Ranked by agronomic viability</span>
              </div>

              {recommendations.map((crop, idx) => {
                const isTop = idx === 0;
                return (
                  <div
                    key={crop.name}
                    className={`bg-white rounded-2xl p-5 border transition shadow-xs ${
                      isTop ? 'border-emerald-300 ring-2 ring-emerald-500/20' : 'border-slate-200'
                    }`}
                  >
                    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center pb-3 border-b border-slate-100 gap-2">
                      <div className="flex items-center space-x-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs ${
                          isTop ? 'bg-emerald-600 text-white shadow' : 'bg-slate-100 text-slate-700'
                        }`}>
                          #{idx + 1}
                        </div>
                        <div>
                          <h4 className="text-base font-extrabold text-slate-900">{crop.name}</h4>
                          <span className="text-[11px] text-slate-500 italic">{crop.scientific_name} • {crop.season}</span>
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${
                          crop.suitability_score >= 82
                            ? 'bg-emerald-100 text-emerald-800'
                            : 'bg-amber-100 text-amber-800'
                        }`}>
                          {crop.suitability_category} ({crop.suitability_score}%)
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 my-3 text-xs">
                      <div className="p-2 bg-slate-50 rounded-lg">
                        <span className="text-[10px] text-slate-400 block font-semibold uppercase">Ideal Temp</span>
                        <span className="font-bold text-slate-800">{crop.ideal_temperature_range}</span>
                      </div>
                      <div className="p-2 bg-slate-50 rounded-lg">
                        <span className="text-[10px] text-slate-400 block font-semibold uppercase">Water Need</span>
                        <span className="font-bold text-slate-800">{crop.water_requirement}</span>
                      </div>
                      <div className="p-2 bg-slate-50 rounded-lg">
                        <span className="text-[10px] text-slate-400 block font-semibold uppercase">Cycle Duration</span>
                        <span className="font-bold text-slate-800">{crop.growing_duration}</span>
                      </div>
                      <div className="p-2 bg-slate-50 rounded-lg">
                        <span className="text-[10px] text-slate-400 block font-semibold uppercase">Irrigation System</span>
                        <span className="font-bold text-emerald-700">{crop.irrigation_method}</span>
                      </div>
                    </div>

                    <div className="space-y-2 text-xs text-slate-600 pt-2 border-t border-slate-100">
                      <p>
                        <strong className="text-slate-800">Soil Requirements:</strong> {crop.preferred_soil_conditions}
                      </p>
                      <p>
                        <strong className="text-slate-800">Fertilizer & Soil Prep:</strong> {crop.fertilizer_guidelines}
                      </p>
                      <p>
                        <strong className="text-emerald-700">Pest & Disease Scouting:</strong> {crop.pest_precautions}
                      </p>
                      <p className="text-amber-800 bg-amber-50/50 p-2 rounded-lg border border-amber-100">
                        <strong>Agronomic Risk Alert:</strong> {crop.risks_limitations}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="bg-slate-50 rounded-2xl p-12 border border-dashed border-slate-200 text-center space-y-3">
              <Sprout className="w-10 h-10 text-emerald-600/40 mx-auto" />
              <h4 className="text-sm font-bold text-slate-700">No Crop Recommendation Generated Yet</h4>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                Set your soil pH, texture, and seasonal water availability on the left and click "Analyze & Rank Crops" to inspect the top 5 crops.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CropRecommender;
