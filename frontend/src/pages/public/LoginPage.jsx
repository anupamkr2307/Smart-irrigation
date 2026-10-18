import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { useAuth, DEMO_ACCOUNTS } from '../../context/AuthContext';
import {
  Droplets, Lock, Mail, ArrowRight, UserCheck, ShieldAlert,
  Sprout, Store, GraduationCap, ShoppingBag, ShieldCheck, CheckCircle2
} from 'lucide-react';
import toast from 'react-hot-toast';

const PORTAL_TABS = [
  { id: 'all', label: 'All Portals (Auto-Detect)', icon: Droplets, color: 'emerald' },
  { id: 'farmer', label: 'Farmer Portal', icon: Sprout, color: 'emerald' },
  { id: 'retailer', label: 'Retailer Desk', icon: Store, color: 'sky' },
  { id: 'specialist', label: 'Specialist Portal', icon: GraduationCap, color: 'indigo' },
  { id: 'customer', label: 'Customer Portal', icon: ShoppingBag, color: 'amber' },
  { id: 'admin', label: 'Admin Console', icon: ShieldCheck, color: 'slate' }
];

const LoginPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialPortal = searchParams.get('portal') || 'all';
  const [targetPortal, setTargetPortal] = useState(initialPortal);

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [permissionError, setPermissionError] = useState(null);

  const { login, quickLoginAs, loading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const qPortal = searchParams.get('portal');
    if (qPortal && qPortal !== targetPortal) {
      setTargetPortal(qPortal);
    }
  }, [searchParams]);

  const handlePortalChange = (newPortal) => {
    setTargetPortal(newPortal);
    setPermissionError(null);
    if (newPortal === 'all') {
      searchParams.delete('portal');
      setSearchParams(searchParams);
    } else {
      setSearchParams({ portal: newPortal });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password) {
      toast.error('Please enter both email and password.');
      return;
    }

    setPermissionError(null);
    const res = await login(email, password, targetPortal);
    if (res.success) {
      toast.success(`Welcome back, ${res.user.full_name}!`);
      navigate(`/${res.user.role}/dashboard`);
    } else {
      if (res.status === 403 && res.user_role) {
        setPermissionError({
          message: res.error,
          userRole: res.user_role,
          targetRole: res.target_role
        });
      } else {
        toast.error(res.error);
      }
    }
  };

  const handleSwitchToCorrectPortal = async () => {
    if (!permissionError) return;
    const correctRole = permissionError.userRole;
    handlePortalChange(correctRole);
    setPermissionError(null);
    toast.loading(`Signing in to ${correctRole.toUpperCase()} Portal...`, { id: 'auto-redirect' });
    const res = await login(email, password, correctRole);
    if (res.success) {
      toast.success(`Welcome to your ${res.user.role} dashboard!`, { id: 'auto-redirect' });
      navigate(`/${res.user.role}/dashboard`);
    } else {
      toast.error(res.error, { id: 'auto-redirect' });
    }
  };

  const handleDemoLogin = async (acc) => {
    setEmail(acc.email);
    setPassword(acc.pass);
    setTargetPortal(acc.role);
    setPermissionError(null);
    toast.loading(`Authenticating as ${acc.role}...`, { id: 'demo-login' });
    const res = await quickLoginAs(acc.role);
    if (res.success) {
      toast.success(`Signed in as ${res.user.full_name}`, { id: 'demo-login' });
      navigate(`/${res.user.role}/dashboard`);
    } else {
      toast.error('Login failed', { id: 'demo-login' });
    }
  };

  return (
    <div className="min-h-[calc(100vh-8rem)] flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 bg-slate-50">
      <div className="max-w-md w-full space-y-6 bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm">
        <div className="text-center">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-emerald-600 to-sky-500 flex items-center justify-center text-white mx-auto shadow-md">
            <Droplets className="w-7 h-7" />
          </div>
          <h2 className="mt-4 text-2xl font-extrabold text-slate-900 tracking-tight">
            Sign In to Smart Irrigation
          </h2>
          <p className="mt-1 text-xs text-slate-500">
            Select your role portal or sign in to auto-detect your authorized access
          </p>
        </div>

        {/* Portal Filter Tabs */}
        <div>
          <label className="block text-[11px] font-bold text-slate-600 uppercase tracking-wider mb-2">
            Target Role Portal:
          </label>
          <div className="grid grid-cols-3 gap-1.5 p-1 bg-slate-100 rounded-xl">
            {PORTAL_TABS.map((tab) => {
              const Icon = tab.icon;
              const isSelected = targetPortal === tab.id;
              return (
                <button
                  type="button"
                  key={tab.id}
                  onClick={() => handlePortalChange(tab.id)}
                  className={`py-1.5 px-2 rounded-lg text-[11px] font-semibold flex items-center justify-center space-x-1.5 transition ${
                    isSelected
                      ? 'bg-white text-emerald-800 shadow-xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5 shrink-0" />
                  <span className="truncate">{tab.label.split(' ')[0]}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Permission Denied Alert Card */}
        {permissionError && (
          <div className="p-4 rounded-xl bg-red-50 border border-red-200 space-y-2">
            <div className="flex items-start space-x-2.5">
              <ShieldAlert className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-xs font-bold text-red-900">Permission Denied</h4>
                <p className="text-xs text-red-700 mt-0.5 leading-relaxed">
                  {permissionError.message}
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={handleSwitchToCorrectPortal}
              className="w-full mt-2 py-2 px-3 bg-red-600 hover:bg-red-700 text-white rounded-lg text-xs font-bold transition flex items-center justify-center space-x-1.5"
            >
              <span>Switch to {permissionError.userRole.toUpperCase()} Portal & Enter</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* One-click Demo Accounts Pill Box */}
        <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200">
          <p className="text-[11px] font-bold text-slate-600 uppercase tracking-wider mb-2 flex items-center space-x-1">
            <UserCheck className="w-3.5 h-3.5 text-emerald-600" />
            <span>Instant Demo Logins (Click to Sign In):</span>
          </p>
          <div className="grid grid-cols-2 gap-2">
            {DEMO_ACCOUNTS.map((acc) => (
              <button
                type="button"
                key={acc.role}
                onClick={() => handleDemoLogin(acc)}
                className="text-left px-2.5 py-1.5 rounded-lg bg-white border border-slate-200 hover:border-emerald-500 hover:bg-emerald-50/50 text-xs font-semibold text-slate-700 transition"
              >
                <span className="block capitalize font-bold text-emerald-700">{acc.role}</span>
                <span className="text-[10px] text-slate-500 truncate block">{acc.email}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-slate-200" />
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-white px-2 text-slate-400 font-medium">Or enter credentials</span>
          </div>
        </div>

        {/* Manual Login Form */}
        <form className="space-y-4" onSubmit={handleSubmit}>
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Email Address</label>
            <div className="relative">
              <Mail className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                placeholder="name@example.com"
                className="w-full pl-9 pr-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">Password</label>
            <div className="relative">
              <Lock className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                placeholder="••••••••"
                className="w-full pl-9 pr-3 py-2 text-sm border border-slate-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 px-4 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-bold rounded-lg shadow-sm transition flex items-center justify-center space-x-2 disabled:opacity-50"
          >
            <span>{loading ? 'Authenticating...' : `Sign In to ${targetPortal === 'all' ? 'Smart Irrigation' : targetPortal.toUpperCase()}`}</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </form>

        <div className="text-center space-y-2 pt-2 border-t border-slate-100">
          <p className="text-xs text-slate-600">
            Need an account for this portal?{' '}
            <Link
              to={targetPortal !== 'all' ? `/register?role=${targetPortal}` : '/register'}
              className="font-bold text-emerald-600 hover:text-emerald-500"
            >
              Open {targetPortal !== 'all' ? targetPortal.toUpperCase() : 'Role'} Registration Portal →
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
