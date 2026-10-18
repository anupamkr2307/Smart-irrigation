import React from 'react';
import { Droplets, Sprout, Store, MessageSquare, Shield, CheckCircle } from 'lucide-react';
import { Link } from 'react-router-dom';

const FeaturesPage = () => {
  const featuresByRole = [
    {
      role: 'Farmer Capabilities',
      icon: Sprout,
      color: 'emerald',
      items: [
        'Multi-plot farm registration with interactive Leaflet GPS pinning',
        'Machine learning irrigation depth (mm) and volume (Liters/m³) predictions',
        'Automatic live weather, humidity, and rainfall forecast fetch via Open-Meteo',
        'Multi-criteria top-5 crop suitability recommendations',
        'Tailored agricultural advisories for pest and soil management',
        'Direct consultation requests with certified agricultural specialists',
        'Order irrigation kits and bulk input price quotations from retailers'
      ]
    },
    {
      role: 'Retailer Tools',
      icon: Store,
      color: 'sky',
      items: [
        'Store and inventory management with real-time stock quantities',
        'Catalog listing across 5 categories (Seeds, Fertilizers, Irrigation, Tools, Bio-inputs)',
        'Incoming order processing with shipment lifecycle tracking',
        'Custom quotation generator responding to farmer bulk requests',
        'Sales analytics and transaction history'
      ]
    },
    {
      role: 'Specialist Workdesk',
      icon: MessageSquare,
      color: 'indigo',
      items: [
        'Consultation triage inbox for crop disease, pest, and irrigation issues',
        'Interactive threaded chat with crop, soil, and plot diagnostic details',
        'Verified advisory badge attached to expert recommendations',
        'Farmer feedback ratings and consultation resolution tracking'
      ]
    },
    {
      role: 'Administrator & System Governance',
      icon: Shield,
      color: 'purple',
      items: [
        'Platform-wide user role and active status management',
        'Specialist credential verification workflow',
        'ML model diagnostics inspecting MAE, RMSE, and R² scores',
        'Live dataset viewer inspecting all 1,500+ agronomic records',
        'Security and administrative audit trail logging'
      ]
    }
  ];

  return (
    <div className="bg-slate-50 min-h-screen py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Platform Features & Capabilities</h1>
          <p className="mt-3 text-sm text-slate-600">
            A comprehensive, connected ecosystem designed to modernize irrigation, protect water resources, and drive rural prosperity.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {featuresByRole.map((section) => {
            const Icon = section.icon;
            return (
              <div key={section.role} className="bg-white rounded-2xl p-7 border border-slate-200 shadow-xs">
                <div className="flex items-center space-x-3 mb-5">
                  <div className="w-10 h-10 rounded-xl bg-slate-100 flex items-center justify-center text-slate-800">
                    <Icon className="w-5 h-5 text-emerald-600" />
                  </div>
                  <h3 className="text-lg font-bold text-slate-900">{section.role}</h3>
                </div>

                <ul className="space-y-3">
                  {section.items.map((item, idx) => (
                    <li key={idx} className="flex items-start space-x-2 text-xs text-slate-600 leading-relaxed">
                      <CheckCircle className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>

        <div className="mt-12 text-center">
          <Link
            to="/register"
            className="inline-flex items-center px-6 py-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl font-bold shadow-md transition"
          >
            Create Your Account Today
          </Link>
        </div>
      </div>
    </div>
  );
};

export default FeaturesPage;
