import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useUnit } from '../../context/UnitContext';
import { ShoppingBag, Truck, CheckCircle2, Clock } from 'lucide-react';
import toast from 'react-hot-toast';

const RetailerOrders = () => {
  const { formatCurrency } = useUnit();
  const [orders, setOrders] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadOrders();
  }, []);

  const loadOrders = async () => {
    setLoading(true);
    try {
      const res = await api.get('/marketplace/orders');
      setOrders(res.data.orders || []);
    } catch (err) {
      console.error(err);
      toast.error('Failed to load incoming orders');
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateStatus = async (orderId, newStatus) => {
    try {
      await api.patch(`/marketplace/orders/${orderId}/status`, { status: newStatus });
      toast.success(`Order marked as ${newStatus}`);
      loadOrders();
    } catch (err) {
      toast.error('Failed to update status');
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs">
        <h1 className="text-2xl font-extrabold text-slate-900">Incoming Orders & Fulfillment</h1>
        <p className="text-xs text-slate-500 mt-1">Review checkout orders, verify dispatch addresses, and update shipment lifecycle.</p>
      </div>

      <div className="space-y-4">
        {orders.length === 0 ? (
          <div className="bg-white rounded-2xl p-12 border border-slate-200 text-center text-xs text-slate-500">
            No incoming customer orders found.
          </div>
        ) : (
          orders.map((o) => (
            <div key={o.id} className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs space-y-4">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center pb-3 border-b border-slate-100 gap-2">
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-xs font-bold text-slate-700">{o.order_number}</span>
                    <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800">
                      {o.status}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">Customer: <strong>{o.customer_name}</strong> • Phone: {o.shipping?.phone}</p>
                </div>

                <div className="text-right">
                  <span className="text-base font-extrabold text-slate-900 block">{formatCurrency(o.total_amount)}</span>
                  <span className="text-[10px] text-emerald-600 font-semibold uppercase">{o.payment_status}</span>
                </div>
              </div>

              {/* Order items */}
              <div className="space-y-2">
                <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider block">Items Purchased:</span>
                {o.items?.map((item) => (
                  <div key={item.id} className="flex justify-between text-xs py-1 px-3 bg-slate-50 rounded-lg">
                    <span>{item.product_name} × {item.quantity}</span>
                    <span className="font-semibold text-slate-800">{formatCurrency(item.subtotal)}</span>
                  </div>
                ))}
              </div>

              {/* Delivery Address and Actions */}
              <div className="pt-3 border-t border-slate-100 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
                <p className="text-[11px] text-slate-600">
                  <strong>Ship to:</strong> {o.shipping?.address}, {o.shipping?.city} - {o.shipping?.pincode}
                </p>

                <div className="flex items-center space-x-2">
                  <select
                    value={o.status}
                    onChange={(e) => handleUpdateStatus(o.id, e.target.value)}
                    className="text-xs px-2.5 py-1.5 border border-slate-300 rounded-lg bg-white font-medium"
                  >
                    <option value="pending">Pending</option>
                    <option value="confirmed">Confirmed</option>
                    <option value="shipped">Shipped</option>
                    <option value="delivered">Delivered</option>
                    <option value="cancelled">Cancelled</option>
                  </select>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default RetailerOrders;
