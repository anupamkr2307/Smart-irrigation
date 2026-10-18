import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useUnit } from '../context/UnitContext';
import { Droplets, Sprout, Menu, X, User, LogOut, LayoutDashboard, ShoppingCart, Sliders } from 'lucide-react';

const Navbar = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const { areaUnit, volumeUnit, toggleAreaUnit, toggleVolumeUnit } = useUnit();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navigate = useNavigate();

  const getDashboardLink = () => {
    if (!user) return '/login';
    return `/${user.role}/dashboard`;
  };

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <nav className="bg-white border-b border-slate-200 sticky top-7 z-40 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo & Brand */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2.5">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-sky-500 flex items-center justify-center text-white shadow-md">
                <Droplets className="w-6 h-6" />
              </div>
              <div>
                <span className="text-lg font-bold text-slate-900 tracking-tight flex items-center">
                  Smart<span className="text-emerald-600">Irrigation</span>
                </span>
                <span className="block text-[10px] text-slate-500 font-medium tracking-wide">
                  Complete Agriculture Platform
                </span>
              </div>
            </Link>

            {/* Desktop Navigation Links */}
            <div className="hidden md:flex ml-10 space-x-6">
              <Link to="/" className="text-slate-600 hover:text-emerald-600 font-medium text-sm transition">
                Home
              </Link>
              <Link to="/about" className="text-slate-600 hover:text-emerald-600 font-medium text-sm transition">
                About FAO-56
              </Link>
              <Link to="/features" className="text-slate-600 hover:text-emerald-600 font-medium text-sm transition">
                Features
              </Link>
              <Link to="/marketplace" className="text-slate-600 hover:text-emerald-600 font-medium text-sm transition">
                Marketplace
              </Link>
              <Link to="/contact" className="text-slate-600 hover:text-emerald-600 font-medium text-sm transition">
                Advisory & Help
              </Link>
            </div>
          </div>

          {/* Right Action Buttons */}
          <div className="hidden md:flex items-center space-x-4">
            {/* Unit Toggles */}
            <div className="flex items-center bg-slate-100 rounded-lg p-1 text-xs font-semibold text-slate-700">
              <button
                onClick={toggleAreaUnit}
                className="px-2 py-1 rounded bg-white shadow-xs hover:bg-slate-50 transition uppercase"
                title="Toggle Area Unit"
              >
                {areaUnit}
              </button>
              <span className="text-slate-300 mx-1">|</span>
              <button
                onClick={toggleVolumeUnit}
                className="px-2 py-1 rounded bg-white shadow-xs hover:bg-slate-50 transition uppercase"
                title="Toggle Volume Unit"
              >
                {volumeUnit}
              </button>
            </div>

            {isAuthenticated ? (
              <div className="flex items-center space-x-3">
                <Link
                  to={getDashboardLink()}
                  className="flex items-center space-x-1.5 px-3 py-1.5 bg-emerald-50 text-emerald-700 hover:bg-emerald-100 rounded-lg text-sm font-semibold transition"
                >
                  <LayoutDashboard className="w-4 h-4" />
                  <span>{user.role.toUpperCase()} Portal</span>
                </Link>

                <div className="flex items-center space-x-2 pl-2 border-l border-slate-200">
                  <div className="text-right">
                    <p className="text-xs font-bold text-slate-800 leading-tight">{user.full_name}</p>
                    <span className="text-[10px] text-slate-500 capitalize">{user.role}</span>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="p-1.5 text-slate-400 hover:text-red-600 rounded-lg transition"
                    title="Sign Out"
                  >
                    <LogOut className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex items-center space-x-2">
                <Link
                  to="/login"
                  className="px-4 py-2 text-sm font-medium text-slate-700 hover:text-emerald-600 transition"
                >
                  Sign In
                </Link>
                <Link
                  to="/register"
                  className="px-4 py-2 text-sm font-medium text-white bg-emerald-600 hover:bg-emerald-700 rounded-lg shadow-sm transition"
                >
                  Get Started
                </Link>
              </div>
            )}
          </div>

          {/* Mobile Menu Button */}
          <div className="flex items-center md:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-md text-slate-600 hover:text-slate-900 focus:outline-none"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-slate-200 bg-white px-4 pt-2 pb-4 space-y-2">
          <Link
            to="/"
            onClick={() => setMobileMenuOpen(false)}
            className="block px-3 py-2 rounded-md text-base font-medium text-slate-700 hover:bg-slate-100"
          >
            Home
          </Link>
          <Link
            to="/about"
            onClick={() => setMobileMenuOpen(false)}
            className="block px-3 py-2 rounded-md text-base font-medium text-slate-700 hover:bg-slate-100"
          >
            About FAO-56
          </Link>
          <Link
            to="/features"
            onClick={() => setMobileMenuOpen(false)}
            className="block px-3 py-2 rounded-md text-base font-medium text-slate-700 hover:bg-slate-100"
          >
            Features
          </Link>
          <Link
            to="/marketplace"
            onClick={() => setMobileMenuOpen(false)}
            className="block px-3 py-2 rounded-md text-base font-medium text-slate-700 hover:bg-slate-100"
          >
            Marketplace
          </Link>
          <Link
            to="/contact"
            onClick={() => setMobileMenuOpen(false)}
            className="block px-3 py-2 rounded-md text-base font-medium text-slate-700 hover:bg-slate-100"
          >
            Advisory & Support
          </Link>

          <div className="pt-4 border-t border-slate-200">
            {isAuthenticated ? (
              <div className="space-y-2">
                <Link
                  to={getDashboardLink()}
                  onClick={() => setMobileMenuOpen(false)}
                  className="block w-full text-center px-4 py-2 bg-emerald-600 text-white rounded-lg font-medium"
                >
                  Go to {user.role.toUpperCase()} Dashboard
                </Link>
                <button
                  onClick={() => {
                    handleLogout();
                    setMobileMenuOpen(false);
                  }}
                  className="block w-full text-center px-4 py-2 bg-slate-100 text-slate-700 rounded-lg font-medium"
                >
                  Sign Out
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-2">
                <Link
                  to="/login"
                  onClick={() => setMobileMenuOpen(false)}
                  className="text-center px-4 py-2 border border-slate-300 rounded-lg text-slate-700 font-medium"
                >
                  Sign In
                </Link>
                <Link
                  to="/register"
                  onClick={() => setMobileMenuOpen(false)}
                  className="text-center px-4 py-2 bg-emerald-600 text-white rounded-lg font-medium"
                >
                  Register
                </Link>
              </div>
            )}
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
