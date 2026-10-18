import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useUnit } from '../../context/UnitContext';
import { FileText, Send, Tag, Clock } from 'lucide-react';
import toast from 'react-hot-toast';

const QuotationDesk = () => {
  const { formatCurrency } = useUnit();
  const [requests, setRequests] = useState([]);
  const [loading, setLoading] = useState(true);

  // Bid form state
  const [activeRequest, setActiveRequest] = useState(null);
  const [offeredPrice, setOfferedPrice] = useState(12000);
  const [deliveryDays, setDeliveryDays] = useState(4);
  const [remarks, setRemarks] = useState('');

  useEffect(() => {
    loadRequests();
  }, []);

  const loadRequests = async () => {
    setLoading(true);
    try {
      const res = await api.get('/marketplace/quotations');
      setRequests(res.data.quotation_requests || []);
    } catch (err) {
      console.error(err);
      toast.error('Failed to load quotation requests');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitQuote = async (e) => {
    e.preventDefault();
    if (!activeRequest || offeredPrice <= 0) return;

    try {
      await api.post(`/marketplace/quotations/${activeRequest.id}/respond`, {
        offered_price: parseFloat(offeredPrice),
        estimated_delivery_days: parseInt(deliveryDays),
        remarks
      });
      toast.success('Your quotation offer was sent to the farmer!');
      setActiveRequest(null);
      setRemarks('');
      loadRequests();
    } catch (err) {
      toast.error('Failed to submit quote');
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs">
        <h1 className="text-2xl font-extrabold text-slate-900">Farmer Bulk Quotation Requests</h1>
        <p className="text-xs text-slate-500 mt-1">Review bulk equipment RFQs submitted by regional farmers and submit competitive price quotes.</p>
      </div>

      <div className="space-y-4">
        {requests.length === 0 ? (
          <div className="bg-white rounded-2xl p-12 border border-slate-200 text-center text-xs text-slate-500">
            No pending bulk quotation requests found.
          </div>
        ) : (
          requests.map((req) => (
            <div key={req.id} className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs space-y-4">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center pb-3 border-b border-slate-100 gap-2">
                <div>
                  <h3 className="text-base font-bold text-slate-900">{req.item_title}</h3>
                  <p className="text-xs text-slate-500">
                    Requested by: <strong>{req.farmer_name}</strong> • Location: <strong>{req.farm_location}</strong>
                  </p>
                </div>
                <div className="text-right">
                  <span className="text-xs font-bold bg-slate-100 text-slate-800 px-2.5 py-1 rounded-md">
                    Quantity: {req.quantity}
                  </span>
                </div>
              </div>

              <p className="text-xs text-slate-600 leading-relaxed">{req.description}</p>

              {/* Existing quotes */}
              {req.quotations?.length > 0 && (
                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
                  <span className="text-[10px] font-bold text-slate-500 uppercase block">Existing Bids:</span>
                  {req.quotations.map((q) => (
                    <div key={q.id} className="flex justify-between items-center text-xs py-1 px-2.5 bg-white rounded border border-slate-200">
                      <span>{q.retailer_name}: {q.remarks} ({q.estimated_delivery_days} days)</span>
                      <strong className="text-emerald-700">{formatCurrency(q.offered_price)}</strong>
                    </div>
                  ))}
                </div>
              )}

              <div className="pt-2 flex justify-end">
                <button
                  onClick={() => {
                    setActiveRequest(req);
                    setOfferedPrice(15000);
                  }}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold shadow-xs transition"
                >
                  Submit Custom Quote
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Quote Submission Modal */}
      {activeRequest && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl p-6 max-w-md w-full space-y-4 shadow-xl border border-slate-200">
            <h3 className="text-base font-bold text-slate-900 pb-2 border-b border-slate-100">
              Submit Quotation for {activeRequest.item_title}
            </h3>

            <form onSubmit={handleSubmitQuote} className="space-y-3">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Total Offered Price (INR) *</label>
                <input
                  type="number"
                  step="10"
                  required
                  value={offeredPrice}
                  onChange={(e) => setOfferedPrice(e.target.value)}
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Estimated Delivery (Days)</label>
                <input
                  type="number"
                  value={deliveryDays}
                  onChange={(e) => setDeliveryDays(e.target.value)}
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Remarks & Specifications</label>
                <textarea
                  rows={3}
                  value={remarks}
                  onChange={(e) => setRemarks(e.target.value)}
                  placeholder="Includes delivery to farm gate, warranty, installation support..."
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div className="flex justify-end space-x-2 pt-2">
                <button
                  type="button"
                  onClick={() => setActiveRequest(null)}
                  className="px-3.5 py-1.5 text-xs text-slate-600"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-xs font-bold"
                >
                  Send Quotation to Farmer
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default QuotationDesk;
