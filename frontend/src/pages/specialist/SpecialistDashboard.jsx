import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { MessageSquare, ShieldCheck, Send, CheckCircle2, User, Clock, AlertTriangle } from 'lucide-react';
import toast from 'react-hot-toast';

const SpecialistDashboard = () => {
  const { user } = useAuth();
  const [consultations, setConsultations] = useState([]);
  const [selectedCase, setSelectedCase] = useState(null);
  const [replyText, setReplyText] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadConsultations();
    const interval = setInterval(loadConsultations, 10000);
    return () => clearInterval(interval);
  }, [statusFilter]);

  const loadConsultations = async () => {
    try {
      const url = statusFilter ? `/consultations?status=${statusFilter}` : '/consultations';
      const res = await api.get(url);
      const list = res.data.consultations || [];
      setConsultations(list);
      if (selectedCase) {
        const updated = list.find((c) => c.id === selectedCase.id);
        if (updated) setSelectedCase(updated);
      } else if (list.length > 0) {
        setSelectedCase(list[0]);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSendAdvice = async (e) => {
    e.preventDefault();
    if (!selectedCase || !replyText.trim()) return;

    try {
      await api.post(`/consultations/${selectedCase.id}/messages`, {
        message: replyText.trim()
      });
      toast.success('Specialist diagnostic advice posted!');
      setReplyText('');
      loadConsultations();
    } catch (err) {
      toast.error('Failed to post advice');
    }
  };

  const handleSetStatus = async (newStatus) => {
    if (!selectedCase) return;
    try {
      await api.patch(`/consultations/${selectedCase.id}/status`, { status: newStatus });
      toast.success(`Case status set to ${newStatus}`);
      loadConsultations();
    } catch (err) {
      toast.error('Failed to update status');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest flex items-center space-x-1">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Specialist Diagnostics & Agronomy Desk</span>
          </span>
          <h1 className="text-2xl font-extrabold text-slate-900 mt-1">Farmer Consultations Inbox</h1>
          <p className="text-xs text-slate-500 mt-1">
            Answer incoming farmer diagnosis requests, verify field symptoms, and prescribe treatment protocols.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
          >
            <option value="">All Consultations</option>
            <option value="open">Open Cases</option>
            <option value="in_progress">In Progress</option>
            <option value="resolved">Resolved</option>
          </select>
        </div>
      </div>

      {/* Main Two-Column Workdesk */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Cases List (4 cols) */}
        <div className="lg:col-span-4 bg-white rounded-2xl p-5 border border-slate-200 shadow-xs space-y-3">
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2">
            Incoming Queries ({consultations.length})
          </h3>

          {consultations.length === 0 ? (
            <p className="text-xs text-slate-400 py-10 text-center">No consultation queries found.</p>
          ) : (
            consultations.map((c) => {
              const isSelected = selectedCase?.id === c.id;
              return (
                <div
                  key={c.id}
                  onClick={() => setSelectedCase(c)}
                  className={`p-3.5 rounded-xl border cursor-pointer transition ${
                    isSelected
                      ? 'bg-emerald-50/70 border-emerald-300 shadow-xs'
                      : 'bg-slate-50 border-slate-200 hover:border-slate-300'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <span className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-slate-200 text-slate-700">
                      {c.issue_category}
                    </span>
                    <span className={`text-[10px] font-semibold capitalize px-2 py-0.5 rounded-full ${
                      c.status === 'resolved'
                        ? 'bg-emerald-100 text-emerald-800'
                        : 'bg-amber-100 text-amber-800'
                    }`}>
                      {c.status.replace('_', ' ')}
                    </span>
                  </div>

                  <h4 className="text-xs font-bold text-slate-900 mt-2 truncate">{c.subject}</h4>
                  <p className="text-[11px] text-slate-500 line-clamp-1 mt-0.5">Farmer: {c.farmer_name} • Crop: {c.crop_name}</p>

                  <div className="flex justify-between items-center text-[10px] text-slate-400 mt-2 pt-2 border-t border-slate-200/60">
                    <span>Soil: {c.soil_type || 'N/A'}</span>
                    <span>{c.messages?.length || 1} messages</span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Diagnosis Chat & Recommendation Area (8 cols) */}
        <div className="lg:col-span-8 bg-white rounded-2xl p-6 border border-slate-200 shadow-xs flex flex-col justify-between min-h-[500px]">
          {selectedCase ? (
            <>
              {/* Header */}
              <div className="pb-4 border-b border-slate-100 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
                <div>
                  <h3 className="text-base font-bold text-slate-900">{selectedCase.subject}</h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Farmer: <strong>{selectedCase.farmer_name}</strong> • Location: <strong>{selectedCase.location || 'Local Farm'}</strong> • Crop: <strong>{selectedCase.crop_name}</strong>
                  </p>
                </div>

                <div className="flex items-center space-x-2">
                  <select
                    value={selectedCase.status}
                    onChange={(e) => handleSetStatus(e.target.value)}
                    className="text-xs px-2.5 py-1.5 border border-slate-300 rounded-lg bg-white"
                  >
                    <option value="open">Open</option>
                    <option value="in_progress">In Progress</option>
                    <option value="resolved">Resolved</option>
                  </select>
                </div>
              </div>

              {/* Chat Thread */}
              <div className="flex-1 overflow-y-auto py-4 space-y-4 max-h-[380px]">
                {selectedCase.image_url && (
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 inline-block mb-2">
                    <p className="text-[10px] font-bold text-slate-500 uppercase mb-1">Farmer's Photo Attachment:</p>
                    <img src={selectedCase.image_url} alt="Crop Issue" className="max-h-48 rounded-lg object-cover" />
                  </div>
                )}

                {selectedCase.messages?.map((msg) => {
                  const isSpecialist = msg.is_specialist_advice;
                  return (
                    <div
                      key={msg.id}
                      className={`flex flex-col ${isSpecialist ? 'items-end' : 'items-start'}`}
                    >
                      <div className="flex items-center space-x-1 mb-1 text-[11px] text-slate-500">
                        <span className="font-bold text-slate-700">{msg.sender_name}</span>
                        {isSpecialist && (
                          <span className="inline-flex items-center space-x-0.5 text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded font-semibold text-[10px]">
                            <ShieldCheck className="w-3 h-3 text-emerald-600" />
                            <span>Specialist Prescription</span>
                          </span>
                        )}
                      </div>

                      <div
                        className={`max-w-lg p-3.5 rounded-2xl text-xs leading-relaxed ${
                          isSpecialist
                            ? 'bg-emerald-600 text-white rounded-br-none shadow-xs'
                            : 'bg-slate-100 text-slate-800 rounded-bl-none'
                        }`}
                      >
                        {msg.message}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Specialist Prescription Box */}
              <form onSubmit={handleSendAdvice} className="pt-3 border-t border-slate-100 space-y-2">
                <textarea
                  rows={3}
                  value={replyText}
                  onChange={(e) => setReplyText(e.target.value)}
                  placeholder="Provide scientific diagnosis, irrigation adjustment, organic or chemical treatment dosage..."
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-xl outline-none focus:ring-2 focus:ring-emerald-500"
                />
                <div className="flex justify-between items-center">
                  <span className="text-[10px] text-emerald-700 font-semibold flex items-center space-x-1">
                    <ShieldCheck className="w-3.5 h-3.5" />
                    <span>Your reply will be badged as Verified Specialist Advice</span>
                  </span>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold shadow-xs transition flex items-center space-x-1"
                  >
                    <Send className="w-4 h-4" />
                    <span>Send Verified Advice</span>
                  </button>
                </div>
              </form>
            </>
          ) : (
            <p className="text-xs text-slate-500 text-center py-20">Select a consultation from the left to diagnose.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default SpecialistDashboard;
