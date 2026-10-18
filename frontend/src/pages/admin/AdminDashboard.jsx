import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useUnit } from '../../context/UnitContext';
import StatCard from '../../components/StatCard';
import { Shield, Users, Droplets, ShoppingBag, Cpu, CheckCircle2, TrendingUp, BarChart2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import toast from 'react-hot-toast';

const AdminDashboard = () => {
  const { formatCurrency, formatVolume } = useUnit();
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    setLoading(true);
    try {
      const res = await api.get('/admin/analytics');
      setAnalytics(res.data);
    } catch (err) {
      console.error(err);
      toast.error('Failed to load admin analytics');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="bg-gradient-to-r from-slate-900 to-indigo-950 rounded-2xl p-6 text-white shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <span className="text-indigo-300 text-xs font-semibold uppercase tracking-wider">
            System Administration & Governance
          </span>
          <h1 className="text-2xl font-bold mt-1">Platform Operations Control</h1>
          <p className="text-xs text-slate-300 mt-1">
            Real-time monitoring of users across 5 roles, ML inference throughput, water conservation metrics, and marketplace commerce.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <Link
            to="/admin/ml-metrics"
            className="px-4 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-slate-950 rounded-xl text-xs font-bold transition flex items-center space-x-1.5"
          >
            <Cpu className="w-4 h-4 text-slate-950" />
            <span>Inspect ML Pipeline</span>
          </Link>
          <Link
            to="/admin/users"
            className="px-4 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl text-xs font-semibold border border-white/20 transition flex items-center space-x-1.5"
          >
            <Users className="w-4 h-4" />
            <span>Manage Accounts</span>
          </Link>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Registered Users"
          value={analytics?.users?.total || 5}
          subtitle={`${analytics?.users?.farmers || 1} Farmers • ${analytics?.users?.specialists || 1} Specialists`}
          icon={Users}
          color="indigo"
        />
        <StatCard
          title="ML Inferences Executed"
          value={analytics?.operations?.irrigation_predictions_run || 0}
          subtitle="Precision water calculations"
          icon={Droplets}
          color="emerald"
        />
        <StatCard
          title="Estimated Water Saved"
          value={formatVolume(analytics?.operations?.estimated_water_saved_liters || 0)}
          subtitle="Vs traditional flood basin"
          icon={TrendingUp}
          color="sky"
        />
        <StatCard
          title="Marketplace Revenue"
          value={formatCurrency(analytics?.operations?.total_marketplace_revenue_inr || 0)}
          subtitle={`${analytics?.operations?.marketplace_orders || 0} completed orders`}
          icon={ShoppingBag}
          color="amber"
        />
      </div>

      {/* Role Breakdown Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs space-y-4">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider pb-2 border-b border-slate-100 flex items-center space-x-2">
            <Users className="w-4 h-4 text-indigo-600" />
            <span>User Distribution by Persona</span>
          </h3>

          <div className="space-y-3 text-xs">
            <div className="flex justify-between items-center p-3 bg-slate-50 rounded-xl">
              <span className="font-semibold text-slate-800">Farmers & Cultivators</span>
              <strong className="text-emerald-700 font-bold">{analytics?.users?.farmers || 1} accounts</strong>
            </div>
            <div className="flex justify-between items-center p-3 bg-slate-50 rounded-xl">
              <span className="font-semibold text-slate-800">Agricultural Retailers & Vendors</span>
              <strong className="text-sky-700 font-bold">{analytics?.users?.retailers || 1} accounts</strong>
            </div>
            <div className="flex justify-between items-center p-3 bg-slate-50 rounded-xl">
              <span className="font-semibold text-slate-800">Certified Specialists & Agronomists</span>
              <strong className="text-indigo-700 font-bold">{analytics?.users?.specialists || 1} accounts</strong>
            </div>
            <div className="flex justify-between items-center p-3 bg-slate-50 rounded-xl">
              <span className="font-semibold text-slate-800">Marketplace Consumers</span>
              <strong className="text-amber-700 font-bold">{analytics?.users?.customers || 1} accounts</strong>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs space-y-4">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider pb-2 border-b border-slate-100 flex items-center space-x-2">
            <Shield className="w-4 h-4 text-emerald-600" />
            <span>Quick Governance Actions</span>
          </h3>

          <div className="space-y-2.5">
            <Link
              to="/admin/users"
              className="block p-3 bg-slate-50 hover:bg-slate-100 rounded-xl border border-slate-200 transition text-xs"
            >
              <strong className="text-slate-900 block font-bold">User Role & Verification Desk</strong>
              <span className="text-slate-500">Approve specialist licenses, grant or deactivate accounts</span>
            </Link>

            <Link
              to="/admin/ml-metrics"
              className="block p-3 bg-slate-50 hover:bg-slate-100 rounded-xl border border-slate-200 transition text-xs"
            >
              <strong className="text-slate-900 block font-bold">ML Diagnostics & Dataset Inspector</strong>
              <span className="text-slate-500">Review Linear Regression coefficients and 1,500 structured records</span>
            </Link>

            <Link
              to="/admin/audit-logs"
              className="block p-3 bg-slate-50 hover:bg-slate-100 rounded-xl border border-slate-200 transition text-xs"
            >
              <strong className="text-slate-900 block font-bold">Security Audit Trail</strong>
              <span className="text-slate-500">Track logins, status changes, order payments, and model inferences</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
