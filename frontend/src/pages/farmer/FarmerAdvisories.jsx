import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { FileText, AlertTriangle, ShieldCheck, Filter } from 'lucide-react';
import toast from 'react-hot-toast';

const FarmerAdvisories = () => {
  const [advisories, setAdvisories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [categoryFilter, setCategoryFilter] = useState('');

  useEffect(() => {
    loadAdvisories();
  }, [categoryFilter]);

  const loadAdvisories = async () => {
    setLoading(true);
    try {
      const url = categoryFilter ? `/advisories?category=${encodeURIComponent(categoryFilter)}` : '/advisories';
      const res = await api.get(url);
      setAdvisories(res.data.advisories || []);
    } catch (err) {
      console.error(err);
      toast.error('Failed to load advisories');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest flex items-center space-x-1">
            <FileText className="w-3.5 h-3.5" />
            <span>Expert Guidance Bulletins</span>
          </span>
          <h1 className="text-2xl font-extrabold text-slate-900 mt-1">Agricultural Advisories</h1>
          <p className="text-xs text-slate-500 mt-1">
            Verified best practices for micro-irrigation scheduling, waterlogging prevention, and soil nutrient conservation.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <Filter className="w-4 h-4 text-slate-400" />
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
          >
            <option value="">All Categories</option>
            <option value="Irrigation">Irrigation</option>
            <option value="Soil Management">Soil Management</option>
            <option value="Water Conservation">Water Conservation</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {advisories.map((adv) => (
          <div
            key={adv.id}
            className={`bg-white rounded-2xl p-6 border transition shadow-xs ${
              adv.is_emergency ? 'border-amber-300 bg-amber-50/20' : 'border-slate-200'
            }`}
          >
            <div className="flex justify-between items-start mb-3">
              <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                {adv.category}
              </span>
              {adv.is_specialist_verified && (
                <span className="flex items-center space-x-1 text-emerald-700 text-[11px] font-semibold bg-emerald-50 px-2 py-0.5 rounded">
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                  <span>Verified Advice</span>
                </span>
              )}
            </div>

            <h3 className="text-base font-bold text-slate-900 mb-2">{adv.title}</h3>
            <p className="text-xs text-slate-600 leading-relaxed mb-4">{adv.advisory_text}</p>

            {adv.precautions && (
              <div className="p-3 bg-amber-50/70 rounded-xl border border-amber-200 text-xs text-amber-900 space-y-1">
                <strong className="block font-semibold">Critical Precautions:</strong>
                <p className="text-[11px] text-amber-800 leading-relaxed">{adv.precautions}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default FarmerAdvisories;
