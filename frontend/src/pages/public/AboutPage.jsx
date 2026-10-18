import React from 'react';
import { Droplets, BookOpen, Cpu, ShieldCheck, Check } from 'lucide-react';

const AboutPage = () => {
  return (
    <div className="bg-slate-50 min-h-screen py-12">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-100 text-emerald-800 text-xs font-semibold mb-3">
            <BookOpen className="w-3.5 h-3.5" />
            <span>Agronomic Science & Methodology</span>
          </div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">The Science Behind Smart Irrigation</h1>
          <p className="mt-2 text-sm text-slate-600">Standardized meteorological physics combined with machine learning pipelines.</p>
        </div>

        {/* Content sections */}
        <div className="space-y-8">
          {/* Section 1 */}
          <div className="bg-white rounded-2xl p-7 border border-slate-200 shadow-xs">
            <h2 className="text-lg font-bold text-slate-900 mb-3 flex items-center space-x-2">
              <Droplets className="w-5 h-5 text-emerald-600" />
              <span>1. FAO-56 Penman-Monteith Reference Evapotranspiration</span>
            </h2>
            <p className="text-xs text-slate-600 leading-relaxed mb-4">
              Our calculation engine implements the universally accepted Food and Agriculture Organization (FAO-56)
              Penman-Monteith equation to estimate reference evapotranspiration ($ET_0$) from net solar radiation,
              temperature, relative humidity, and wind speed.
            </p>
            <div className="bg-slate-900 text-emerald-300 p-4 rounded-xl font-mono text-xs overflow-x-auto leading-relaxed">
              ET0 = [ 0.408·Δ·(Rn - G) + γ·(900 / (T + 273))·u2·(es - ea) ] / [ Δ + γ·(1 + 0.34·u2) ]
            </div>
            <p className="text-[11px] text-slate-500 mt-2">
              Where $Rn$ is net radiation, $T$ is temperature in °C, $u2$ is wind speed at 2m height, and $(es - ea)$ represents vapor pressure deficit.
            </p>
          </div>

          {/* Section 2 */}
          <div className="bg-white rounded-2xl p-7 border border-slate-200 shadow-xs">
            <h2 className="text-lg font-bold text-slate-900 mb-3 flex items-center space-x-2">
              <Cpu className="w-5 h-5 text-sky-600" />
              <span>2. Scikit-Learn Machine Learning Pipeline</span>
            </h2>
            <p className="text-xs text-slate-600 leading-relaxed mb-3">
              We employ a scikit-learn <code className="bg-slate-100 text-slate-800 px-1.5 py-0.5 rounded font-mono text-xs">Pipeline</code> trained
              on 1,500 validated agro-climatic records across 15 agricultural districts.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 my-4">
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-center">
                <span className="text-xs text-slate-500 font-medium">Algorithm</span>
                <p className="text-sm font-bold text-slate-900 mt-1">Linear Regression</p>
              </div>
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-center">
                <span className="text-xs text-slate-500 font-medium">Evaluation Metric</span>
                <p className="text-sm font-bold text-slate-900 mt-1">MAE: 1.78 mm</p>
              </div>
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-center">
                <span className="text-xs text-slate-500 font-medium">Held-Out Test R²</span>
                <p className="text-sm font-bold text-slate-900 mt-1">0.70</p>
              </div>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              Categorical inputs (Soil type, Crop type, Growth stage) are encoded via <code className="bg-slate-100 px-1 py-0.5 rounded font-mono">OneHotEncoder</code>,
              and numerical features are standardized through <code className="bg-slate-100 px-1 py-0.5 rounded font-mono">StandardScaler</code>.
              Physical constraints ensure predictions are non-negative and properly bounded by soil field capacity.
            </p>
          </div>

          {/* Section 3 */}
          <div className="bg-white rounded-2xl p-7 border border-slate-200 shadow-xs">
            <h2 className="text-lg font-bold text-slate-900 mb-3 flex items-center space-x-2">
              <ShieldCheck className="w-5 h-5 text-indigo-600" />
              <span>3. Water Balance & Physical Conversions</span>
            </h2>
            <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
              <div className="flex items-start space-x-2">
                <Check className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
                <p><strong>Net Irrigation Requirement:</strong> Net Irrigation (mm) = max(0, Crop Water Requirement − Effective Rain − Usable Soil Water)</p>
              </div>
              <div className="flex items-start space-x-2">
                <Check className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
                <p><strong>Gross Irrigation Accounting:</strong> Gross Depth (mm) = Net Irrigation / Application Efficiency (Drip: 90%, Sprinkler: 75%, Furrow: 60%)</p>
              </div>
              <div className="flex items-start space-x-2">
                <Check className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
                <p><strong>Volume Conversion:</strong> 1 mm depth across 1 m² = exactly 1 Liter. (1 Acre = 4,046.86 m², 1 Hectare = 10,000 m²)</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;
