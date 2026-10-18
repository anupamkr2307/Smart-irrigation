import React from 'react';
import { NavLink } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  LayoutDashboard,
  Sprout,
  Droplets,
  Layers,
  MessageSquare,
  ShoppingBag,
  Package,
  Store,
  FileText,
  UserCheck,
  Shield,
  BarChart3,
  Users,
  Settings
} from 'lucide-react';

const Sidebar = () => {
  const { user } = useAuth();
  if (!user) return null;

  const role = user.role;

  const farmerNav = [
    { name: 'Dashboard Overview', path: '/farmer/dashboard', icon: LayoutDashboard },
    { name: 'Farms & Plots', path: '/farmer/farms', icon: Layers },
    { name: 'Irrigation Calculator', path: '/farmer/irrigation', icon: Droplets },
    { name: 'Crop Recommender', path: '/farmer/crops', icon: Sprout },
    { name: 'Advisory Bulletins', path: '/farmer/advisories', icon: FileText },
    { name: 'Specialist Consultation', path: '/farmer/consultations', icon: MessageSquare },
    { name: 'Marketplace & Orders', path: '/farmer/marketplace', icon: ShoppingBag },
  ];

  const retailerNav = [
    { name: 'Store Dashboard', path: '/retailer/dashboard', icon: LayoutDashboard },
    { name: 'Product Inventory', path: '/retailer/inventory', icon: Package },
    { name: 'Customer Orders', path: '/retailer/orders', icon: ShoppingBag },
    { name: 'Farmer Quotations', path: '/retailer/quotations', icon: FileText },
  ];

  const customerNav = [
    { name: 'Catalog & Store', path: '/customer/dashboard', icon: Store },
    { name: 'My Orders', path: '/customer/orders', icon: ShoppingBag },
  ];

  const specialistNav = [
    { name: 'Consultation Desk', path: '/specialist/dashboard', icon: MessageSquare },
    { name: 'Advisory Publisher', path: '/specialist/publish-advisory', icon: FileText },
    { name: 'Professional Profile', path: '/specialist/profile', icon: UserCheck },
  ];

  const adminNav = [
    { name: 'Admin Dashboard', path: '/admin/dashboard', icon: LayoutDashboard },
    { name: 'User Management', path: '/admin/users', icon: Users },
    { name: 'ML Pipeline & Data', path: '/admin/ml-metrics', icon: BarChart3 },
    { name: 'Security Audit Logs', path: '/admin/audit-logs', icon: Shield },
  ];

  let items = farmerNav;
  if (role === 'retailer') items = retailerNav;
  if (role === 'customer') items = customerNav;
  if (role === 'specialist') items = specialistNav;
  if (role === 'admin') items = adminNav;

  return (
    <aside className="w-64 bg-white border-r border-slate-200 min-h-[calc(100vh-6rem)] p-4 flex flex-col justify-between hidden md:flex">
      <div>
        <div className="mb-6 px-3 py-2 bg-slate-50 rounded-xl border border-slate-100 flex items-center space-x-3">
          <div className="w-10 h-10 rounded-full bg-emerald-100 text-emerald-800 flex items-center justify-center font-bold text-sm">
            {user.full_name?.charAt(0) || 'U'}
          </div>
          <div className="overflow-hidden">
            <h4 className="text-sm font-bold text-slate-800 truncate">{user.full_name}</h4>
            <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase bg-emerald-100 text-emerald-800">
              {user.role}
            </span>
          </div>
        </div>

        <p className="px-3 text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2">
          Management Portal
        </p>

        <nav className="space-y-1">
          {items.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm font-medium transition ${
                    isActive
                      ? 'bg-emerald-600 text-white shadow-xs'
                      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                  }`
                }
              >
                <Icon className="w-4 h-4 flex-shrink-0" />
                <span>{item.name}</span>
              </NavLink>
            );
          })}
        </nav>
      </div>

      <div className="pt-4 border-t border-slate-200 text-xs text-slate-500 text-center">
        <span>Smart Irrigation v1.0</span>
      </div>
    </aside>
  );
};

export default Sidebar;
