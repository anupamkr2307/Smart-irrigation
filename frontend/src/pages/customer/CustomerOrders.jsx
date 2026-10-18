import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useUnit } from '../../context/UnitContext';
import { ShoppingBag, Truck, CheckCircle2, Clock } from 'lucide-react';
import toast from 'react-hot-toast';

const CustomerOrders = () => {
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
      toast.error('Failed to load orders');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs">
        <h1 className="text-2xl font-extrabold text-slate-900">My Purchase History & Tracking</h1>
        <p className="text-xs text-slate-500 mt-1">Track dispatch status, payment receipts, and delivery timelines.</p>
      </div>

      <div className="space-y-4">
        {orders.length === 0 ? (
          <div className="bg-white rounded-2xl p-12 border border-slate-200 text-center text-xs text-slate-500">
            No orders placed yet. Visit the catalog to order inputs.
          </div>
        ) : (
          orders.map((o) => (
            <div key={o.id} className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs space-y-4">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center pb-3 border-b border-slate-100 gap-2">
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-xs font-bold text-slate-800">{o.order_number}</span>
                    <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800">
                      {o.status}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500 mt-0.5">Purchased from: {o.retailer_name}</p>
                </div>

                <div className="text-right">
                  <span className="text-base font-extrabold text-slate-900 block">{formatCurrency(o.total_amount)}</span>
                  <span className="text-[10px] text-emerald-600 font-semibold uppercase">{o.payment_method}</span>
                </div>
              </div>

              {/* Items */}
              <div className="space-y-2">
                {o.items?.map((item) => (
                  <div key={item.id} className="flex justify-between text-xs py-1 px-3 bg-slate-50 rounded-lg">
                    <span>{item.product_name} × {item.quantity}</span>
                    <span className="font-semibold">{formatCurrency(item.subtotal)}</span>
                  </div>
                ))}
              </div>

              <div className="pt-2 text-[11px] text-slate-500 flex justify-between items-center">
                <span>Shipping to: {o.shipping?.name}, {o.shipping?.address}, {o.shipping?.city}</span>
                <span className="text-slate-400">Order ID: #{o.id}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default CustomerOrders;
