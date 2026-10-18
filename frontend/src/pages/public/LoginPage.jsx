import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth, DEMO_ACCOUNTS } from '../../context/AuthContext';
import { Droplets, Lock, Mail, ArrowRight, UserCheck } from 'lucide-react';
import toast from 'react-hot-toast';

const LoginPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { login, quickLoginAs, loading } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password) {
      toast.error('Please enter both email and password.');
      return;
    }

    const res = await login(email, password);
    if (res.success) {
      toast.success(`Welcome back, ${res.user.full_name}!`);
      navigate(`/${res.user.role}/dashboard`);
    } else {
      toast.error(res.error);
    }
  };

  const handleDemoLogin = async (acc) => {
    setEmail(acc.email);
    setPassword(acc.pass);
    toast.loading(`Logging in as ${acc.role}...`, { id: 'demo-login' });
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
      <div className="max-w-md w-full space-y-8 bg-white p-8 rounded-2xl border border-slate-200 shadow-sm">
        <div className="text-center">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-emerald-600 to-sky-500 flex items-center justify-center text-white mx-auto shadow-md">
            <Droplets className="w-7 h-7" />
          </div>
          <h2 className="mt-4 text-2xl font-extrabold text-slate-900 tracking-tight">
            Sign In to Smart Irrigation
          </h2>
          <p className="mt-1 text-xs text-slate-500">
            Access your agricultural insights, water predictions, and marketplace
          </p>
        </div>

        {/* One-click Demo Accounts Pill Box */}
        <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200">
          <p className="text-[11px] font-bold text-slate-600 uppercase tracking-wider mb-2 flex items-center space-x-1">
            <UserCheck className="w-3.5 h-3.5 text-emerald-600" />
            <span>Instant Demo Logins (Click to Enter):</span>
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
        <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
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
            <span>{loading ? 'Authenticating...' : 'Sign In'}</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </form>

        <p className="text-center text-xs text-slate-600">
          Don't have an account yet?{' '}
          <Link to="/register" className="font-semibold text-emerald-600 hover:text-emerald-500">
            Create an account
          </Link>
        </p>
      </div>
    </div>
  );
};

export default LoginPage;
