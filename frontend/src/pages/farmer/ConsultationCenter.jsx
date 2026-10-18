import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import {
  MessageSquare,
  Send,
  PlusCircle,
  CheckCircle,
  Star,
  ShieldCheck,
  User,
  Clock,
  AlertCircle
} from 'lucide-react';
import toast from 'react-hot-toast';

const ConsultationCenter = () => {
  const { user } = useAuth();
  const [consultations, setConsultations] = useState([]);
  const [selectedConsultation, setSelectedConsultation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showNewModal, setShowNewModal] = useState(false);

  // New consultation form state
  const [subject, setSubject] = useState('');
  const [issueCategory, setIssueCategory] = useState('Pest & Disease');
  const [cropName, setCropName] = useState('Tomato');
  const [soilType, setSoilType] = useState('loamy');
  const [description, setDescription] = useState('');
  const [imageUrl, setImageUrl] = useState('');

  // Reply message
  const [replyText, setReplyText] = useState('');
  // Rating modal
  const [ratingVal, setRatingVal] = useState(5);
  const [feedbackText, setFeedbackText] = useState('');
  const [showRateModal, setShowRateModal] = useState(false);

  useEffect(() => {
    loadConsultations();
    // Poll every 10 seconds for real-time conversation updates
    const interval = setInterval(loadConsultations, 10000);
    return () => clearInterval(interval);
  }, []);

  const loadConsultations = async () => {
    try {
      const res = await api.get('/consultations');
      const list = res.data.consultations || [];
      setConsultations(list);
      if (selectedConsultation) {
        const updated = list.find((c) => c.id === selectedConsultation.id);
        if (updated) setSelectedConsultation(updated);
      } else if (list.length > 0) {
        setSelectedConsultation(list[0]);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!subject || !description) {
      toast.error('Subject and description are required.');
      return;
    }

    try {
      const res = await api.post('/consultations', {
        subject,
        issue_category: issueCategory,
        crop_name: cropName,
        soil_type: soilType,
        description,
        image_url: imageUrl || null
      });
      toast.success('Consultation submitted to specialists!');
      setShowNewModal(false);
      setSubject('');
      setDescription('');
      setImageUrl('');
      loadConsultations();
    } catch (err) {
      console.error(err);
      toast.error('Failed to submit consultation request');
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!selectedConsultation || !replyText.trim()) return;

    try {
      await api.post(`/consultations/${selectedConsultation.id}/messages`, {
        message: replyText.trim()
      });
      setReplyText('');
      loadConsultations();
    } catch (err) {
      toast.error('Failed to send message');
    }
  };

  const handleRate = async (e) => {
    e.preventDefault();
    if (!selectedConsultation) return;

    try {
      await api.post(`/consultations/${selectedConsultation.id}/rate`, {
        rating: ratingVal,
        feedback: feedbackText
      });
      toast.success('Thank you for rating your specialist consultation!');
      setShowRateModal(false);
      loadConsultations();
    } catch (err) {
      toast.error('Failed to submit rating');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest flex items-center space-x-1">
            <MessageSquare className="w-3.5 h-3.5" />
            <span>Specialist Diagnostic Desk</span>
          </span>
          <h1 className="text-2xl font-extrabold text-slate-900 mt-1">Agricultural Consultations</h1>
          <p className="text-xs text-slate-500 mt-1">
            Directly ask questions to certified agronomists and soil specialists regarding pest attacks, leaf discoloration, and irrigation issues.
          </p>
        </div>

        <button
          onClick={() => setShowNewModal(true)}
          className="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold shadow-xs transition flex items-center space-x-1.5"
        >
          <PlusCircle className="w-4 h-4" />
          <span>Ask Specialist Question</span>
        </button>
      </div>

      {/* Main Grid: Consultation Threads & Conversation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left List (4 cols) */}
        <div className="lg:col-span-4 bg-white rounded-2xl p-5 border border-slate-200 shadow-xs space-y-3">
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2">
            My Consultations ({consultations.length})
          </h3>

          {consultations.length === 0 ? (
            <div className="text-center py-10 text-slate-400 text-xs">
              No questions submitted yet. Click above to ask.
            </div>
          ) : (
            consultations.map((c) => {
              const isSelected = selectedConsultation?.id === c.id;
              return (
                <div
                  key={c.id}
                  onClick={() => setSelectedConsultation(c)}
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
                        : c.status === 'in_progress'
                        ? 'bg-sky-100 text-sky-800'
                        : 'bg-amber-100 text-amber-800'
                    }`}>
                      {c.status.replace('_', ' ')}
                    </span>
                  </div>

                  <h4 className="text-xs font-bold text-slate-900 mt-2 truncate">{c.subject}</h4>
                  <p className="text-[11px] text-slate-500 line-clamp-1 mt-0.5">{c.description}</p>

                  <div className="flex justify-between items-center text-[10px] text-slate-400 mt-2 pt-2 border-t border-slate-200/60">
                    <span>Crop: {c.crop_name || 'General'}</span>
                    <span>{c.messages?.length || 1} messages</span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Right Conversation Window (8 cols) */}
        <div className="lg:col-span-8 bg-white rounded-2xl p-6 border border-slate-200 shadow-xs flex flex-col justify-between min-h-[500px]">
          {selectedConsultation ? (
            <>
              {/* Header Info */}
              <div className="pb-4 border-b border-slate-100 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
                <div>
                  <div className="flex items-center space-x-2">
                    <h3 className="text-base font-bold text-slate-900">{selectedConsultation.subject}</h3>
                    <span className="text-xs bg-slate-100 text-slate-700 px-2 py-0.5 rounded font-medium">
                      {selectedConsultation.issue_category}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    Crop: <strong>{selectedConsultation.crop_name || 'N/A'}</strong> • Soil: <strong>{selectedConsultation.soil_type || 'N/A'}</strong>
                    {selectedConsultation.specialist_name && (
                      <span className="ml-2 text-emerald-700 font-semibold flex-inline items-center space-x-1">
                        • Assigned Specialist: {selectedConsultation.specialist_name}
                      </span>
                    )}
                  </p>
                </div>

                {selectedConsultation.status !== 'resolved' && (
                  <button
                    onClick={() => setShowRateModal(true)}
                    className="px-3 py-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-800 rounded-lg text-xs font-semibold border border-emerald-200 transition"
                  >
                    Mark Resolved & Rate
                  </button>
                )}
              </div>

              {/* Message List */}
              <div className="flex-1 overflow-y-auto py-4 space-y-4 max-h-[400px]">
                {selectedConsultation.image_url && (
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 inline-block mb-2">
                    <p className="text-[10px] font-bold text-slate-500 uppercase mb-1">Attached Crop Photo:</p>
                    <img
                      src={selectedConsultation.image_url}
                      alt="Crop Issue"
                      className="max-h-48 rounded-lg object-cover"
                    />
                  </div>
                )}

                {selectedConsultation.messages?.map((msg) => {
                  const isMe = msg.sender_id === user?.id;
                  const isSpecialistAdvice = msg.is_specialist_advice;
                  return (
                    <div
                      key={msg.id}
                      className={`flex flex-col ${isMe ? 'items-end' : 'items-start'}`}
                    >
                      <div className="flex items-center space-x-1 mb-1 text-[11px] text-slate-500">
                        <span className="font-bold text-slate-700">{msg.sender_name}</span>
                        {isSpecialistAdvice && (
                          <span className="inline-flex items-center space-x-0.5 text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded font-semibold text-[10px]">
                            <ShieldCheck className="w-3 h-3 text-emerald-600" />
                            <span>Verified Specialist Advice</span>
                          </span>
                        )}
                      </div>

                      <div
                        className={`max-w-lg p-3.5 rounded-2xl text-xs leading-relaxed ${
                          isMe
                            ? 'bg-emerald-600 text-white rounded-br-none'
                            : isSpecialistAdvice
                            ? 'bg-emerald-50 border border-emerald-200 text-slate-800 rounded-bl-none shadow-xs'
                            : 'bg-slate-100 text-slate-800 rounded-bl-none'
                        }`}
                      >
                        {msg.message}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Reply Box */}
              <form onSubmit={handleSendMessage} className="pt-3 border-t border-slate-100 flex items-center space-x-2">
                <input
                  type="text"
                  value={replyText}
                  onChange={(e) => setReplyText(e.target.value)}
                  placeholder="Type follow-up response or question..."
                  className="flex-1 px-4 py-2.5 text-xs border border-slate-300 rounded-xl outline-none focus:ring-2 focus:ring-emerald-500"
                />
                <button
                  type="submit"
                  className="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold shadow-xs transition flex items-center space-x-1"
                >
                  <Send className="w-4 h-4" />
                  <span>Send</span>
                </button>
              </form>
            </>
          ) : (
            <p className="text-xs text-slate-500 text-center py-20">Select a consultation from the left to view the thread.</p>
          )}
        </div>
      </div>

      {/* New Consultation Modal */}
      {showNewModal && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl p-6 max-w-lg w-full space-y-4 shadow-xl border border-slate-200 animate-fade-in">
            <h3 className="text-base font-bold text-slate-900 pb-2 border-b border-slate-100">
              Submit Question to Agricultural Specialist
            </h3>

            <form onSubmit={handleCreate} className="space-y-3">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Issue Subject *</label>
                <input
                  type="text"
                  required
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  placeholder="e.g. Yellowing margins on tomato leaves and wilting"
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Category</label>
                  <select
                    value={issueCategory}
                    onChange={(e) => setIssueCategory(e.target.value)}
                    className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white"
                  >
                    <option value="Pest & Disease">Pest & Disease</option>
                    <option value="Soil & Nutrition">Soil & Nutrition</option>
                    <option value="Irrigation Scheduling">Irrigation Scheduling</option>
                    <option value="Weather Damage">Weather / Heat Damage</option>
                    <option value="General Advisory">General Advisory</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Crop Name</label>
                  <input
                    type="text"
                    value={cropName}
                    onChange={(e) => setCropName(e.target.value)}
                    placeholder="e.g. Tomato"
                    className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Problem Description *</label>
                <textarea
                  rows={4}
                  required
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Describe symptoms, irrigation frequency, when it started, and fertilizers applied..."
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Photo Attachment URL (Optional)</label>
                <input
                  type="url"
                  value={imageUrl}
                  onChange={(e) => setImageUrl(e.target.value)}
                  placeholder="https://images.unsplash.com/... or image link"
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div className="flex justify-end space-x-2 pt-3">
                <button
                  type="button"
                  onClick={() => setShowNewModal(false)}
                  className="px-4 py-2 border border-slate-300 text-slate-600 rounded-lg text-xs font-semibold"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-bold"
                >
                  Submit Question
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Rating Modal */}
      {showRateModal && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl p-6 max-w-md w-full space-y-4 shadow-xl border border-slate-200">
            <h3 className="text-base font-bold text-slate-900 text-center">
              Rate Consultation with Specialist
            </h3>

            <div className="flex justify-center space-x-2 py-3">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  type="button"
                  key={star}
                  onClick={() => setRatingVal(star)}
                  className="p-1 focus:outline-none"
                >
                  <Star
                    className={`w-7 h-7 ${
                      star <= ratingVal ? 'text-amber-400 fill-amber-400' : 'text-slate-300'
                    }`}
                  />
                </button>
              ))}
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Feedback / Notes</label>
              <textarea
                rows={3}
                value={feedbackText}
                onChange={(e) => setFeedbackText(e.target.value)}
                placeholder="Was the specialist guidance helpful and actionable?"
                className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
              />
            </div>

            <div className="flex justify-end space-x-2 pt-2">
              <button
                type="button"
                onClick={() => setShowRateModal(false)}
                className="px-3 py-1.5 text-xs text-slate-600"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleRate}
                className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-xs font-bold"
              >
                Submit Rating & Close Case
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ConsultationCenter;
