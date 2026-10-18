import React from 'react';
import { Droplets, Heart, Shield, Leaf, PhoneCall, Mail } from 'lucide-react';
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer className="bg-slate-900 text-slate-300 pt-12 pb-8 border-t border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-10">
          {/* Brand & mission */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2.5">
              <div className="w-9 h-9 rounded-lg bg-emerald-600 flex items-center justify-center text-white">
                <Droplets className="w-5 h-5" />
              </div>
              <span className="text-lg font-bold text-white tracking-tight">
                Smart<span className="text-emerald-500">Irrigation</span>
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              AI-powered agricultural lifecycle management platform empowering farmers with
              machine learning water predictions, FAO-56 irrigation calculations, and an integrated rural marketplace.
            </p>
            <div className="flex items-center space-x-3 text-xs text-slate-400">
              <span className="flex items-center space-x-1">
                <Shield className="w-3.5 h-3.5 text-emerald-400" />
                <span>Verified Specialist Guidance</span>
              </span>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-sm font-semibold text-white uppercase tracking-wider mb-3">Platform Navigation</h4>
            <ul className="space-y-2 text-xs">
              <li><Link to="/about" className="hover:text-emerald-400 transition">About FAO-56 Science</Link></li>
              <li><Link to="/features" className="hover:text-emerald-400 transition">System Capabilities</Link></li>
              <li><Link to="/marketplace" className="hover:text-emerald-400 transition">Agricultural Marketplace</Link></li>
              <li><Link to="/login" className="hover:text-emerald-400 transition">Sign In to Dashboard</Link></li>
              <li><Link to="/register" className="hover:text-emerald-400 transition">Farmer & Partner Registration</Link></li>
            </ul>
          </div>

          {/* Supported Roles */}
          <div>
            <h4 className="text-sm font-semibold text-white uppercase tracking-wider mb-3">Integrated Ecosystem</h4>
            <ul className="space-y-2 text-xs text-slate-400">
              <li><strong className="text-slate-200">Farmers:</strong> Farm plotting, ML water prediction, crop selection</li>
              <li><strong className="text-slate-200">Specialists:</strong> Crop diagnosis, pest control, scientific advice</li>
              <li><strong className="text-slate-200">Retailers:</strong> Inventory management, bulk quotations</li>
              <li><strong className="text-slate-200">Customers:</strong> Direct procurement and verified reviews</li>
              <li><strong className="text-slate-200">Administrators:</strong> Platform auditing and ML evaluation</li>
            </ul>
          </div>

          {/* Support & Helpline */}
          <div>
            <h4 className="text-sm font-semibold text-white uppercase tracking-wider mb-3">Kisan Support Desk</h4>
            <div className="space-y-3 text-xs text-slate-400">
              <p className="flex items-center space-x-2">
                <PhoneCall className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                <span>Toll-Free Agri Helpline: 1800-180-1551</span>
              </p>
              <p className="flex items-center space-x-2">
                <Mail className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                <span>support@smartirrigation.org</span>
              </p>
              <div className="pt-2">
                <span className="inline-block bg-slate-800 text-emerald-400 text-[11px] font-medium px-2.5 py-1 rounded">
                  Open-Meteo & FAO-56 Standard
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="border-t border-slate-800 pt-6 flex flex-col sm:flex-row justify-between items-center text-xs text-slate-500">
          <p>© {new Date().getFullYear()} Smart Irrigation Platform. All rights reserved.</p>
          <p className="mt-2 sm:mt-0 flex items-center space-x-1">
            <span>Built with precision agronomy and AI for water conservation</span>
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
