import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { useUnit } from '../../context/UnitContext';
import StatCard from '../../components/StatCard';
import { Store, Package, ShoppingBag, FileText, ArrowRight, Plus } from 'lucide-react';
import { Link } from 'react-router-dom';

const RetailerDashboard = () => {
  const { user } = useAuth();
  const { formatCurrency } = useUnit();
  const [products, setProducts] = useState([]);
  const [orders, setOrders] = useState([]);
  const [quotes, setQuotes] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRetailerData();
  }, []);

  const loadRetailerData = async () => {
    setLoading(true);
    try {
      const [prodRes, orderRes, quoteRes] = await Promise.all([
        api.get(`/marketplace/products?retailer_id=${user.id}`).catch(() => ({ data: { products: [] } })),
        api.get('/marketplace/orders').catch(() => ({ data: { orders: [] } })),
        api.get('/marketplace/quotations').catch(() => ({ data: { quotation_requests: [] } }))
      ]);

      setProducts(prodRes.data.products || []);
      setOrders(orderRes.data.orders || []);
      setQuotes(quoteRes.data.quotation_requests || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const totalSales = orders
    .filter((o) => o.payment_status === 'mock_paid' || o.status === 'delivered')
    .reduce((sum, o) => sum + o.total_amount, 0);

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="bg-gradient-to-r from-sky-800 to-slate-900 rounded-2xl p-6 text-white shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <span className="text-sky-300 text-xs font-semibold uppercase tracking-wider">
            Retailer Commerce & Distribution
          </span>
          <h1 className="text-2xl font-bold mt-1">
            {user?.profile?.shop_name || 'Agro Supply Hub'}
          </h1>
          <p className="text-xs text-slate-300 mt-1">
            Manage your agricultural inventory, process incoming customer orders, and bid on farmer bulk requests.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <Link
            to="/retailer/inventory"
            className="px-4 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-slate-950 rounded-xl text-xs font-bold transition flex items-center space-x-1.5"
          >
            <Plus className="w-4 h-4 text-slate-950" />
            <span>Add New Product</span>
          </Link>
          <Link
            to="/retailer/quotations"
            className="px-4 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl text-xs font-semibold border border-white/20 transition flex items-center space-x-1.5"
          >
            <FileText className="w-4 h-4" />
            <span>View Farmer Quotes</span>
          </Link>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Active Products"
          value={products.length}
          subtitle="Listed in catalog"
          icon={Package}
          color="emerald"
        />
        <StatCard
          title="Total Orders"
          value={orders.length}
          subtitle="Customer & farmer checkouts"
          icon={ShoppingBag}
          color="sky"
        />
        <StatCard
          title="Gross Sales Revenue"
          value={formatCurrency(totalSales)}
          subtitle="Processed via platform"
          icon={Store}
          color="amber"
        />
        <StatCard
          title="Farmer Quotes"
          value={quotes.length}
          subtitle="Bulk RFQs awaiting bids"
          icon={FileText}
          color="indigo"
        />
      </div>

      {/* Two Column Layout: Recent Orders & Inventory Health */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Orders */}
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <ShoppingBag className="w-4 h-4 text-sky-600" />
              <span>Incoming Customer Orders</span>
            </h3>
            <Link to="/retailer/orders" className="text-xs text-emerald-600 font-semibold hover:underline">
              View All
            </Link>
          </div>

          {orders.length === 0 ? (
            <p className="text-xs text-slate-500 text-center py-8">No orders received yet.</p>
          ) : (
            <div className="space-y-3">
              {orders.slice(0, 4).map((o) => (
                <div key={o.id} className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 flex justify-between items-center">
                  <div>
                    <span className="text-[10px] font-mono text-slate-500 block">{o.order_number}</span>
                    <h5 className="text-xs font-bold text-slate-900 mt-0.5">{o.customer_name}</h5>
                    <span className="text-[10px] text-slate-400">{o.shipping?.city} • {o.items?.length || 1} items</span>
                  </div>
                  <div className="text-right">
                    <span className="text-xs font-bold text-slate-900 block">{formatCurrency(o.total_amount)}</span>
                    <span className="text-[10px] uppercase font-semibold text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded-full inline-block mt-0.5">
                      {o.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Current Inventory Snapshot */}
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <Package className="w-4 h-4 text-emerald-600" />
              <span>Inventory & Stock Overview</span>
            </h3>
            <Link to="/retailer/inventory" className="text-xs text-emerald-600 font-semibold hover:underline">
              Manage Products
            </Link>
          </div>

          {products.length === 0 ? (
            <p className="text-xs text-slate-500 text-center py-8">No products listed. Add your first item!</p>
          ) : (
            <div className="space-y-3">
              {products.slice(0, 4).map((p) => (
                <div key={p.id} className="p-3 bg-slate-50 rounded-xl border border-slate-200 flex justify-between items-center">
                  <div>
                    <h5 className="text-xs font-bold text-slate-900">{p.name}</h5>
                    <span className="text-[10px] text-slate-500">{p.category_name} • {formatCurrency(p.price)} / {p.unit}</span>
                  </div>
                  <div className="text-right">
                    <span className={`text-xs font-bold block ${p.stock_quantity < 10 ? 'text-amber-600' : 'text-slate-800'}`}>
                      {p.stock_quantity} in stock
                    </span>
                    <span className="text-[10px] text-slate-400">{p.stock_quantity > 0 ? 'Available' : 'Out'}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RetailerDashboard;
