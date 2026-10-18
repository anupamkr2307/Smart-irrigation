import React from 'react';
import { useAuth, DEMO_ACCOUNTS } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import { Shield, UserCheck, Sprout, Store, ShoppingCart, LogOut } from 'lucide-react';
import toast from 'react-hot-toast';

const roleIcons = {
  farmer: Sprout,
  retailer: Store,
  customer: ShoppingCart,
  specialist: UserCheck,
  admin: Shield,
};

const QuickDemoBar = () => {
  const { user, quickLoginAs, logout } = useAuth();
  const navigate = useNavigate();

  const handleRoleSwitch = async (roleName) => {
    toast.loading(`Switching to ${roleName}...`, { id: 'role-switch' });
    const res = await quickLoginAs(roleName);
    if (res?.success) {
      toast.success(`Logged in as ${res.user.full_name} (${roleName.toUpperCase()})`, { id: 'role-switch' });
      navigate(`/${roleName}/dashboard`);
    } else {
      toast.error('Failed to switch role', { id: 'role-switch' });
    }
  };

  return (
    <div className="bg-slate-900 text-white text-xs py-1.5 px-4 flex flex-wrap items-center justify-between border-b border-slate-800 sticky top-0 z-50">
      <div className="flex items-center space-x-2">
        <span className="bg-emerald-600 px-2 py-0.5 rounded font-bold tracking-wider uppercase text-[10px]">
          Demo Access
        </span>
        <span className="hidden sm:inline text-slate-300">
          Switch test persona with one click:
        </span>
      </div>

      <div className="flex items-center space-x-1 sm:space-x-2 overflow-x-auto py-1">
        {DEMO_ACCOUNTS.map((acc) => {
          const Icon = roleIcons[acc.role] || Sprout;
          const isActive = user?.role === acc.role;
          return (
            <button
              key={acc.role}
              onClick={() => handleRoleSwitch(acc.role)}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded transition-all font-medium ${
                isActive
                  ? 'bg-emerald-500 text-slate-950 shadow font-semibold'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span className="capitalize">{acc.role}</span>
            </button>
          );
        })}

        {user && (
          <button
            onClick={() => {
              logout();
              toast.success('Logged out');
              navigate('/');
            }}
            className="flex items-center space-x-1 px-2 py-1 bg-red-900/60 hover:bg-red-800 text-red-200 rounded transition ml-2"
            title="Log Out"
          >
            <LogOut className="w-3.5 h-3.5" />
            <span className="hidden md:inline">Logout</span>
          </button>
        )}
      </div>
    </div>
  );
};

export default QuickDemoBar;
