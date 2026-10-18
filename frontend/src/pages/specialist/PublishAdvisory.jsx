import React, { useState } from 'react';
import api from '../../services/api';
import { FileText, Send, AlertTriangle, ShieldCheck } from 'lucide-react';
import toast from 'react-hot-toast';

const PublishAdvisory = () => {
  const [title, setTitle] = useState('');
  const [category, setCategory] = useState('Irrigation');
  const [targetCrop, setTargetCrop] = useState('');
  const [targetSoil, setTargetSoil] = useState('');
  const [advisoryText, setAdvisoryText] = useState('');
  const [precautions, setPrecautions] = useState('');
  const [isEmergency, setIsEmergency] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!title || !advisoryText) {
      toast.error('Title and advisory text are required.');
      return;
    }

    setSubmitting(true);
    try {
      await api.post('/advisories', {
        title,
        category,
        target_crop: targetCrop || null,
        target_soil: targetSoil || null,
        advisory_text: advisoryText,
        precautions,
        is_emergency: isEmergency
      });
      toast.success('Advisory bulletin published to all farmers!');
      setTitle('');
      setAdvisoryText('');
      setPrecautions('');
      setIsEmergency(false);
    } catch (err) {
      toast.error(err.response?.data?.error || 'Failed to publish advisory');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs">
        <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest flex items-center space-x-1">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>Official Extension Advisory</span>
        </span>
        <h1 className="text-2xl font-extrabold text-slate-900 mt-1">Publish Agricultural Bulletin</h1>
        <p className="text-xs text-slate-500 mt-1">
          Broadcast scientific irrigation alerts, pest outbreak warnings, or extreme weather precautions directly to farmers.
        </p>
      </div>

      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Advisory Headline / Title *</label>
            <input
              type="text"
              required
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Managing Micro-Drip Intervals during 40°C Heatwaves"
              className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg outline-none focus:ring-2 focus:ring-emerald-500"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Category</label>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white"
              >
                <option value="Irrigation">Irrigation Scheduling</option>
                <option value="Soil Management">Soil Management</option>
                <option value="Water Conservation">Water Conservation</option>
                <option value="Pest & Disease">Pest & Disease</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Target Crop (Optional)</label>
              <input
                type="text"
                value={targetCrop}
                onChange={(e) => setTargetCrop(e.target.value)}
                placeholder="All crops or e.g. Tomato"
                className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Target Soil (Optional)</label>
              <select
                value={targetSoil}
                onChange={(e) => setTargetSoil(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white"
              >
                <option value="">All Soil Types</option>
                <option value="sandy">Sandy</option>
                <option value="loamy">Loamy</option>
                <option value="clay">Clay</option>
                <option value="sandy loam">Sandy Loam</option>
                <option value="clay loam">Clay Loam</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Detailed Scientific Guidance *</label>
            <textarea
              rows={4}
              required
              value={advisoryText}
              onChange={(e) => setAdvisoryText(e.target.value)}
              placeholder="State the agronomic reasoning, recommended irrigation timing, fertilizer adjustments, and expected benefit..."
              className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Precautions & Safety Restrictions</label>
            <textarea
              rows={2}
              value={precautions}
              onChange={(e) => setPrecautions(e.target.value)}
              placeholder="Dosage limitations, timing restrictions (avoid midday watering), or root health cautions..."
              className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
            />
          </div>

          <div className="flex items-center space-x-2 pt-2">
            <input
              type="checkbox"
              id="emergency"
              checked={isEmergency}
              onChange={(e) => setIsEmergency(e.target.checked)}
              className="w-4 h-4 text-amber-600 rounded"
            />
            <label htmlFor="emergency" className="text-xs font-semibold text-slate-700 flex items-center space-x-1">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
              <span>Mark as High-Priority / Weather Alert</span>
            </label>
          </div>

          <div className="pt-3 border-t border-slate-100 flex justify-end">
            <button
              type="submit"
              disabled={submitting}
              className="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold shadow-xs transition flex items-center space-x-1.5 disabled:opacity-50"
            >
              <Send className="w-4 h-4" />
              <span>{submitting ? 'Publishing...' : 'Publish Advisory to Platform'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default PublishAdvisory;
