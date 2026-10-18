import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth, DEMO_ACCOUNTS } from '../../context/AuthContext';
import {
  Droplets,
  Sprout,
  Cpu,
  CloudSun,
  ShieldCheck,
  ShoppingBag,
  ArrowRight,
  CheckCircle2,
  TrendingUp,
  BarChart2
} from 'lucide-react';

const LandingPage = () => {
  const { quickLoginAs } = useAuth();

  return (
    <div className="bg-slate-50 min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-b from-emerald-900 via-emerald-800 to-slate-900 text-white pt-20 pb-28">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="text-center max-w-3xl mx-auto">
            <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-emerald-500/20 border border-emerald-400/30 text-emerald-300 text-xs font-semibold mb-6">
              <Cpu className="w-3.5 h-3.5" />
              <span>AI-Powered Irrigation & FAO-56 Agronomy Engine</span>
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight leading-tight">
              Smarter Water for <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-sky-300">Bigger Harvests</span>
            </h1>

            <p className="mt-6 text-base sm:text-lg text-slate-300 leading-relaxed">
              Empowering farmers with machine learning irrigation predictions, real-time weather analytics,
              agronomic crop recommendations, certified specialist consultations, and an integrated rural marketplace.
            </p>

            {/* Quick Demo Launch Buttons */}
            <div className="mt-10 flex flex-wrap justify-center gap-3">
              <Link
                to="/farmer/irrigation"
                className="px-6 py-3.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold shadow-lg transition flex items-center space-x-2"
              >
                <Droplets className="w-5 h-5 text-slate-950" />
                <span>Try Irrigation Calculator</span>
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                to="/marketplace"
                className="px-6 py-3.5 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-white font-semibold border border-slate-700 transition flex items-center space-x-2"
              >
                <ShoppingBag className="w-5 h-5 text-emerald-400" />
                <span>Browse Marketplace</span>
              </Link>
            </div>

            {/* Fast Persona Switcher Pills */}
            <div className="mt-8 pt-8 border-t border-emerald-700/40">
              <p className="text-xs text-emerald-200/80 font-medium uppercase tracking-wider mb-3">
                Or jump immediately into a pre-configured role dashboard:
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {DEMO_ACCOUNTS.map((acc) => (
                  <button
                    key={acc.role}
                    onClick={() => quickLoginAs(acc.role)}
                    className="px-3 py-1.5 rounded-lg bg-emerald-950/60 hover:bg-emerald-600/50 border border-emerald-500/30 text-xs font-semibold text-emerald-200 transition"
                  >
                    Enter as {acc.role.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Decorative background glow */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
      </section>

      {/* Key Stats Bar */}
      <section className="bg-white border-y border-slate-200 py-8 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div>
              <p className="text-3xl font-extrabold text-emerald-600">30–45%</p>
              <p className="text-xs font-medium text-slate-600 mt-1">Water Saved via Precision Drip</p>
            </div>
            <div>
              <p className="text-3xl font-extrabold text-sky-600">1,500+</p>
              <p className="text-xs font-medium text-slate-600 mt-1">Agronomic Dataset Records</p>
            </div>
            <div>
              <p className="text-3xl font-extrabold text-indigo-600">FAO-56</p>
              <p className="text-xs font-medium text-slate-600 mt-1">Standardized Water Balance</p>
            </div>
            <div>
              <p className="text-3xl font-extrabold text-emerald-600">5 Roles</p>
              <p className="text-xs font-medium text-slate-600 mt-1">Connected Agriculture Ecosystem</p>
            </div>
          </div>
        </div>
      </section>

      {/* Core Features Grid */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-16">
            <h2 className="text-xs font-bold text-emerald-600 uppercase tracking-widest mb-2">Integrated Platform</h2>
            <h3 className="text-3xl font-bold text-slate-900 tracking-tight">Everything You Need Across the Crop Lifecycle</h3>
            <p className="mt-3 text-sm text-slate-600">From seed selection to harvest irrigation scheduling, market quotations, and specialist consultations.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="bg-white rounded-2xl p-7 border border-slate-200 shadow-xs hover:shadow-md transition">
              <div className="w-12 h-12 rounded-xl bg-emerald-100 text-emerald-600 flex items-center justify-center mb-5">
                <Cpu className="w-6 h-6" />
              </div>
              <h4 className="text-lg font-bold text-slate-900 mb-2">Machine Learning Irrigation</h4>
              <p className="text-xs text-slate-600 leading-relaxed">
                Trained Linear Regression pipeline evaluating soil texture, root zone moisture, forecast rainfall, and evaporative demand to determine exact millimeter and liter requirements.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="bg-white rounded-2xl p-7 border border-slate-200 shadow-xs hover:shadow-md transition">
              <div className="w-12 h-12 rounded-xl bg-sky-100 text-sky-600 flex items-center justify-center mb-5">
                <CloudSun className="w-6 h-6" />
              </div>
              <h4 className="text-lg font-bold text-slate-900 mb-2">Live Weather & Elevation Maps</h4>
              <p className="text-xs text-slate-600 leading-relaxed">
                Automated Open-Meteo integration fetching temperature, humidity, wind, solar radiation, and 7-day precipitation forecasts with interactive Leaflet farm plot pinning.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="bg-white rounded-2xl p-7 border border-slate-200 shadow-xs hover:shadow-md transition">
              <div className="w-12 h-12 rounded-xl bg-amber-100 text-amber-700 flex items-center justify-center mb-5">
                <Sprout className="w-6 h-6" />
              </div>
              <h4 className="text-lg font-bold text-slate-900 mb-2">Multi-Criteria Crop Suitability</h4>
              <p className="text-xs text-slate-600 leading-relaxed">
                Transparent agronomic engine ranking top-5 suitable crops based on soil pH, moisture capacity, seasonal temperature windows, and local water supply availability.
              </p>
            </div>

            {/* Feature 4 */}
            <div className="bg-white rounded-2xl p-7 border border-slate-200 shadow-xs hover:shadow-md transition">
              <div className="w-12 h-12 rounded-xl bg-indigo-100 text-indigo-600 flex items-center justify-center mb-5">
                <ShieldCheck className="w-6 h-6" />
              </div>
              <h4 className="text-lg font-bold text-slate-900 mb-2">Farmer–Specialist Tele-Advisory</h4>
              <p className="text-xs text-slate-600 leading-relaxed">
                Direct Q&A desk connecting farmers with verified agricultural scientists and agronomists to diagnose pests, soil deficiency, and irrigation stress with image support.
              </p>
            </div>

            {/* Feature 5 */}
            <div className="bg-white rounded-2xl p-7 border border-slate-200 shadow-xs hover:shadow-md transition">
              <div className="w-12 h-12 rounded-xl bg-rose-100 text-rose-600 flex items-center justify-center mb-5">
                <ShoppingBag className="w-6 h-6" />
              </div>
              <h4 className="text-lg font-bold text-slate-900 mb-2">Input Marketplace & Quotations</h4>
              <p className="text-xs text-slate-600 leading-relaxed">
                Source certified seeds, inline drip equipment, organic bio-nutrients, and request bulk price quotations directly from verified rural retail vendors.
              </p>
            </div>

            {/* Feature 6 */}
            <div className="bg-white rounded-2xl p-7 border border-slate-200 shadow-xs hover:shadow-md transition">
              <div className="w-12 h-12 rounded-xl bg-teal-100 text-teal-600 flex items-center justify-center mb-5">
                <BarChart2 className="w-6 h-6" />
              </div>
              <h4 className="text-lg font-bold text-slate-900 mb-2">Audited Admin & ML Analytics</h4>
              <p className="text-xs text-slate-600 leading-relaxed">
                Full transparency dashboard inspecting MAE, RMSE, R² scores, feature coefficients, 1,200+ record dataset viewer, and security action logs.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Role Personas Overview */}
      <section className="bg-slate-100 py-16 border-t border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <h3 className="text-2xl font-bold text-slate-900">One Integrated Platform for 5 Key Stakeholders</h3>
            <p className="text-xs text-slate-600 mt-2">Click below to experience the platform from any user perspective.</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
            {DEMO_ACCOUNTS.map((acc) => (
              <div key={acc.role} className="bg-white rounded-xl p-5 border border-slate-200 shadow-xs flex flex-col justify-between">
                <div>
                  <span className="inline-block px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-slate-100 text-slate-800 mb-2">
                    {acc.role}
                  </span>
                  <h5 className="font-bold text-slate-900 text-sm">{acc.label}</h5>
                  <p className="text-[11px] text-slate-500 font-mono mt-1">{acc.email}</p>
                </div>
                <button
                  onClick={() => quickLoginAs(acc.role)}
                  className="mt-4 w-full py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-semibold transition"
                >
                  Log In as {acc.role.toUpperCase()}
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
