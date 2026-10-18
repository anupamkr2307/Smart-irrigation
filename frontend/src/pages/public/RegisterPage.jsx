import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import {
  Droplets, User, Mail, Lock, Phone, MapPin, Store, Sprout,
  Briefcase, ShieldCheck, CheckCircle2, ArrowRight, Sparkles,
  HelpCircle, GraduationCap, ShoppingBag, Building2, Layers
} from 'lucide-react';
import toast from 'react-hot-toast';

const PORTAL_CONFIG = {
  farmer: {
    id: 'farmer',
    name: 'Farmer Portal',
    badge: 'Agricultural Producer',
    color: 'emerald',
    icon: Sprout,
    themeClasses: {
      cardActive: 'border-emerald-500 bg-emerald-50/40 text-emerald-900 ring-2 ring-emerald-500/30',
      badge: 'bg-emerald-100 text-emerald-800',
      accentText: 'text-emerald-700',
      btn: 'bg-emerald-600 hover:bg-emerald-700 text-white',
    },
    description: 'For cultivators, orchardists, and farm managers seeking precision irrigation, ML crop matching, and specialist agronomic advice.',
    permissions: [
      { name: 'Access to Farmer Control Hub', desc: 'Manage your plots, crops, and live soil health' },
      { name: 'FAO-56 Irrigation Engine', desc: 'Predict exact water volume (Liters/m³) via Penman-Monteith AI' },
      { name: 'Crop Suitability Recommender', desc: 'Rank top-5 viable crops based on soil pH, texture & temperature' },
      { name: '1-on-1 Specialist Tele-Advisory', desc: 'Upload crop pest photos and receive certified advice' },
      { name: 'Bulk Equipment RFQs', desc: 'Post quotation requests to local retailers for bulk drip discounts' },
      { name: 'Direct Marketplace Ordering', desc: 'Order certified seeds, fertilizers, and drip systems' }
    ]
  },
  retailer: {
    id: 'retailer',
    name: 'Retailer Portal',
    badge: 'Agri-Input Merchant',
    color: 'sky',
    icon: Store,
    themeClasses: {
      cardActive: 'border-sky-500 bg-sky-50/40 text-sky-900 ring-2 ring-sky-500/30',
      badge: 'bg-sky-100 text-sky-800',
      accentText: 'text-sky-700',
      btn: 'bg-sky-600 hover:bg-sky-700 text-white',
    },
    description: 'For distributors, local agri-clinics, and drip equipment suppliers expanding their commercial reach to verified local growers.',
    permissions: [
      { name: 'Store & Inventory Management', desc: 'List drip kits, bio-fertilizers, seeds, and sensors' },
      { name: 'Live Stock Control', desc: 'Automated atomic stock verification prevents overselling' },
      { name: 'Farmer Quotation Desk', desc: 'Review bulk RFQs and submit competitive pricing bids' },
      { name: 'Fulfillment & Order Tracking', desc: 'Manage orders, delivery status, and customer invoices' },
      { name: 'Retailer Sales Analytics', desc: 'Inspect regional equipment demand and revenue metrics' }
    ]
  },
  specialist: {
    id: 'specialist',
    name: 'Specialist Portal',
    badge: 'Certified Agronomist',
    color: 'indigo',
    icon: GraduationCap,
    themeClasses: {
      cardActive: 'border-indigo-500 bg-indigo-50/40 text-indigo-900 ring-2 ring-indigo-500/30',
      badge: 'bg-indigo-100 text-indigo-800',
      accentText: 'text-indigo-700',
      btn: 'bg-indigo-600 hover:bg-indigo-700 text-white',
    },
    description: 'For certified agronomists, soil scientists, university researchers, and extension specialists providing diagnostic advisory.',
    permissions: [
      { name: 'Tele-Advisory Diagnostic Desk', desc: 'Review farmer queries with symptom photos and soil data' },
      { name: 'Certified Specialist Responses', desc: 'Submit authenticated diagnostic recommendations' },
      { name: 'Regional Agricultural Advisories', desc: 'Publish broadcast alerts for weather anomalies & pest outbreaks' },
      { name: 'Specialist Accreditation & Rating', desc: 'Earn verified specialist badges and farmer resolution ratings' }
    ]
  },
  customer: {
    id: 'customer',
    name: 'Customer Portal',
    badge: 'Produce & Agri-Buyer',
    color: 'amber',
    icon: ShoppingBag,
    themeClasses: {
      cardActive: 'border-amber-500 bg-amber-50/40 text-amber-900 ring-2 ring-amber-500/30',
      badge: 'bg-amber-100 text-amber-800',
      accentText: 'text-amber-700',
      btn: 'bg-amber-600 hover:bg-amber-700 text-white',
    },
    description: 'For retail produce buyers, food businesses, and home gardeners purchasing verified equipment and farm produce.',
    permissions: [
      { name: 'Full Marketplace Access', desc: 'Browse verified agricultural products and farming supplies' },
      { name: 'Integrated Cart & Checkout', desc: 'Instant order placement with simulated payment verification' },
      { name: 'Order History & Tracking', desc: 'Live dispatch notifications and digital purchase receipts' }
    ]
  },
  admin: {
    id: 'admin',
    name: 'Administrator Portal',
    badge: 'Governance & Security',
    color: 'slate',
    icon: ShieldCheck,
    themeClasses: {
      cardActive: 'border-slate-800 bg-slate-100 text-slate-900 ring-2 ring-slate-800/30',
      badge: 'bg-slate-200 text-slate-800',
      accentText: 'text-slate-800',
      btn: 'bg-slate-900 hover:bg-slate-800 text-white',
    },
    description: 'For platform system operators managing security, RBAC role assignments, ML model drift, and system-wide audits.',
    permissions: [
      { name: 'Full RBAC Access (All Roles)', desc: 'Manage farmer, retailer, specialist, and customer privileges' },
      { name: 'Specialist Verification Toggle', desc: 'Review qualifications and grant official verified badge' },
      { name: 'ML Pipeline Model Diagnostics', desc: 'Inspect MAE, RMSE, R² scores, and retrain pipelines' },
      { name: 'System Security Audit Logging', desc: 'Track all logins, registrations, and data modifications' }
    ]
  }
};

const RegisterPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialRole = searchParams.get('role') || 'farmer';
  const [role, setRole] = useState(PORTAL_CONFIG[initialRole] ? initialRole : 'farmer');

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    full_name: '',
    phone: '',
    // Farmer fields
    farm_name: '',
    district: 'Nashik',
    state: 'Maharashtra',
    total_area: '5.0',
    area_unit: 'acres',
    soil_type: 'loamy',
    primary_crops: 'Tomato, Wheat, Onion',
    preferred_language: 'English',
    // Retailer fields
    shop_name: '',
    gst_number: '',
    license_number: '',
    address: '',
    city: 'Nashik',
    delivery_supported: true,
    // Specialist fields
    specialization: 'General Agronomy',
    qualification: 'M.Sc. Agriculture',
    years_experience: 5,
    organization: 'Mahatma Phule Krishi Vidyapeeth',
    bio: '',
    // Customer fields
    customer_type: 'Individual Farmer / Buyer',
    pincode: '422001',
    // Admin fields
    admin_secret: ''
  });

  const { register, loading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const qRole = searchParams.get('role');
    if (qRole && PORTAL_CONFIG[qRole] && qRole !== role) {
      setRole(qRole);
    }
  }, [searchParams]);

  const handleRoleSelect = (newRole) => {
    setRole(newRole);
    setSearchParams({ role: newRole });
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.email || !formData.password || !formData.full_name) {
      toast.error('Please complete all required fields.');
      return;
    }

    if (role === 'admin' && !formData.admin_secret) {
      toast.error('Admin Master Passkey is required to register as Administrator.');
      return;
    }

    const payload = { ...formData, role };
    const res = await register(payload);
    if (res.success) {
      toast.success(`Welcome to the ${PORTAL_CONFIG[role].name}, ${res.user.full_name}!`);
      navigate(`/${res.user.role}/dashboard`);
    } else {
      toast.error(res.error);
    }
  };

  const currentPortal = PORTAL_CONFIG[role];

  return (
    <div className="min-h-screen py-10 px-4 sm:px-6 lg:px-8 bg-slate-50">
      <div className="max-w-6xl mx-auto">
        {/* Header Section */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-800 text-xs font-semibold mb-3">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Role-Based Multi-Portal Registration</span>
          </div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight sm:text-4xl">
            Join the Smart Irrigation Ecosystem
          </h1>
          <p className="mt-2 text-sm text-slate-600 max-w-2xl mx-auto">
            Select your specialized agricultural role portal. Your account will be provisioned with dedicated permissions, role-specific tools, and tailored analytics.
          </p>
        </div>

        {/* 5-Role Portal Selector Directory */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-8">
          {Object.values(PORTAL_CONFIG).map((p) => {
            const Icon = p.icon;
            const isSelected = role === p.id;
            return (
              <button
                type="button"
                key={p.id}
                onClick={() => handleRoleSelect(p.id)}
                className={`p-3.5 rounded-xl border text-left transition relative flex flex-col justify-between ${
                  isSelected
                    ? p.themeClasses.cardActive
                    : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/80 text-slate-700'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className={`p-2 rounded-lg ${isSelected ? 'bg-white shadow-xs' : 'bg-slate-100'}`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    {isSelected && (
                      <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    )}
                  </div>
                  <h3 className="text-sm font-bold leading-tight">{p.name}</h3>
                  <p className="text-[11px] text-slate-500 font-medium mt-0.5">{p.badge}</p>
                </div>
                <div className="mt-3 pt-2 border-t border-slate-100">
                  <span className="text-[10px] font-semibold text-slate-400">
                    {isSelected ? 'Active Portal' : 'Switch Portal →'}
                  </span>
                </div>
              </button>
            );
          })}
        </div>

        {/* Main Content Grid: Form (Left) & Permissions Preview (Right) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Form Column */}
          <div className="lg:col-span-7 bg-white p-6 sm:p-8 rounded-2xl border border-slate-200 shadow-sm">
            <div className="flex items-center space-x-3 pb-5 mb-5 border-b border-slate-100">
              <div className="p-2.5 rounded-xl bg-slate-100 text-slate-800">
                <currentPortal.icon className="w-6 h-6" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <h2 className="text-xl font-bold text-slate-900">{currentPortal.name} Registration</h2>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${currentPortal.themeClasses.badge}`}>
                    {currentPortal.badge}
                  </span>
                </div>
                <p className="text-xs text-slate-500 mt-0.5">{currentPortal.description}</p>
              </div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Common Account Fields */}
              <div>
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">1. Account Credentials</h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">Full Name *</label>
                    <div className="relative">
                      <User className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                      <input
                        type="text"
                        name="full_name"
                        value={formData.full_name}
                        onChange={handleChange}
                        required
                        placeholder="e.g. Ramesh Patel"
                        className="w-full pl-9 pr-3 py-2 text-sm border border-slate-300 rounded-lg outline-none focus:ring-2 focus:ring-emerald-500"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">Email Address *</label>
                    <div className="relative">
                      <Mail className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                      <input
                        type="email"
                        name="email"
                        value={formData.email}
                        onChange={handleChange}
                        required
                        placeholder="ramesh@example.com"
                        className="w-full pl-9 pr-3 py-2 text-sm border border-slate-300 rounded-lg outline-none focus:ring-2 focus:ring-emerald-500"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">Create Password *</label>
                    <div className="relative">
                      <Lock className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                      <input
                        type="password"
                        name="password"
                        value={formData.password}
                        onChange={handleChange}
                        required
                        placeholder="At least 6 characters"
                        className="w-full pl-9 pr-3 py-2 text-sm border border-slate-300 rounded-lg outline-none focus:ring-2 focus:ring-emerald-500"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">Mobile Phone</label>
                    <div className="relative">
                      <Phone className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                      <input
                        type="tel"
                        name="phone"
                        value={formData.phone}
                        onChange={handleChange}
                        placeholder="+91 98765 43210"
                        className="w-full pl-9 pr-3 py-2 text-sm border border-slate-300 rounded-lg outline-none focus:ring-2 focus:ring-emerald-500"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Dynamic Role-Specific Profiles */}
              <div className="pt-2">
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
                  2. {currentPortal.badge} Profile Details
                </h4>

                {/* Farmer Profile Fields */}
                {role === 'farmer' && (
                  <div className="p-4 bg-emerald-50/50 rounded-xl border border-emerald-100 space-y-3">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">Farm / Estate Name</label>
                        <input
                          type="text"
                          name="farm_name"
                          value={formData.farm_name}
                          onChange={handleChange}
                          placeholder="Sahyadri Agro Farms"
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                        />
                      </div>
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">Primary District</label>
                        <input
                          type="text"
                          name="district"
                          value={formData.district}
                          onChange={handleChange}
                          placeholder="e.g. Nashik"
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">Total Land Area</label>
                        <div className="flex">
                          <input
                            type="number"
                            step="0.5"
                            name="total_area"
                            value={formData.total_area}
                            onChange={handleChange}
                            className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-l-lg bg-white"
                          />
                          <span className="px-2 py-1.5 bg-slate-100 text-slate-600 text-xs border border-l-0 border-slate-300 rounded-r-lg font-semibold">
                            Acres
                          </span>
                        </div>
                      </div>

                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">Soil Texture Class</label>
                        <select
                          name="soil_type"
                          value={formData.soil_type}
                          onChange={handleChange}
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                        >
                          <option value="loamy">Medium Loam</option>
                          <option value="clay">Heavy Clay (Black Cotton)</option>
                          <option value="sandy">Sandy Soil</option>
                          <option value="sandy loam">Sandy Loam</option>
                          <option value="clay loam">Clay Loam</option>
                          <option value="silty">Silty Alluvium</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">Advisory Language</label>
                        <select
                          name="preferred_language"
                          value={formData.preferred_language}
                          onChange={handleChange}
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                        >
                          <option value="English">English</option>
                          <option value="Hindi">हिंदी (Hindi)</option>
                          <option value="Marathi">मराठी (Marathi)</option>
                          <option value="Gujarati">ગુજરાતી (Gujarati)</option>
                        </select>
                      </div>
                    </div>

                    <div>
                      <label className="block text-[11px] font-semibold text-slate-700 mb-1">Current / Planned Crops</label>
                      <input
                        type="text"
                        name="primary_crops"
                        value={formData.primary_crops}
                        onChange={handleChange}
                        placeholder="e.g. Tomato, Wheat, Sugarcane, Onion"
                        className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                      />
                    </div>
                  </div>
                )}

                {/* Retailer Profile Fields */}
                {role === 'retailer' && (
                  <div className="p-4 bg-sky-50/50 rounded-xl border border-sky-100 space-y-3">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">Store / Business Name *</label>
                        <input
                          type="text"
                          name="shop_name"
                          value={formData.shop_name}
                          onChange={handleChange}
                          placeholder="Kisan Seva Kendra & Agri Supplies"
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                          required
                        />
                      </div>
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">GSTIN (15 Digits)</label>
                        <input
                          type="text"
                          name="gst_number"
                          value={formData.gst_number}
                          onChange={handleChange}
                          placeholder="27AAAAA0000A1Z5"
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white font-mono"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">Trade / Fertilizer License</label>
                        <input
                          type="text"
                          name="license_number"
                          value={formData.license_number}
                          onChange={handleChange}
                          placeholder="LIC/AGRI/2026/089"
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white font-mono"
                        />
                      </div>
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">City / Town</label>
                        <input
                          type="text"
                          name="city"
                          value={formData.city}
                          onChange={handleChange}
                          placeholder="Nashik"
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                        />
                      </div>
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">State</label>
                        <input
                          type="text"
                          name="state"
                          value={formData.state}
                          onChange={handleChange}
                          placeholder="Maharashtra"
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                        />
                      </div>
                    </div>

                    <div className="flex items-center space-x-2 pt-1">
                      <input
                        type="checkbox"
                        id="delivery_supported"
                        name="delivery_supported"
                        checked={formData.delivery_supported}
                        onChange={handleChange}
                        className="rounded text-sky-600 focus:ring-sky-500"
                      />
                      <label htmlFor="delivery_supported" className="text-xs text-slate-700 font-medium">
                        Our store supports local farm delivery for bulk orders
                      </label>
                    </div>
                  </div>
                )}

                {/* Specialist Profile Fields */}
                {role === 'specialist' && (
                  <div className="p-4 bg-indigo-50/50 rounded-xl border border-indigo-100 space-y-3">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">Field of Specialization *</label>
                        <select
                          name="specialization"
                          value={formData.specialization}
                          onChange={handleChange}
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                        >
                          <option value="Agronomy & Crop Physiology">Agronomy & Crop Physiology</option>
                          <option value="Soil Physics & Nutrient Management">Soil Physics & Nutrient Management</option>
                          <option value="Micro-Irrigation & Water Engineering">Micro-Irrigation & Water Engineering</option>
                          <option value="Plant Pathology & Disease Diagnosis">Plant Pathology & Disease Diagnosis</option>
                          <option value="Entomology & Pest Control">Entomology & Pest Control</option>
                          <option value="Horticulture & Greenhouse Tech">Horticulture & Greenhouse Tech</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">Highest Degree / Qualification *</label>
                        <input
                          type="text"
                          name="qualification"
                          value={formData.qualification}
                          onChange={handleChange}
                          placeholder="e.g. Ph.D. Agronomy, M.Sc. Soil Science"
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                          required
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">Research Org / University</label>
                        <input
                          type="text"
                          name="organization"
                          value={formData.organization}
                          onChange={handleChange}
                          placeholder="ICAR / State Agricultural Univ"
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                        />
                      </div>

                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">License / Accreditation ID</label>
                        <input
                          type="text"
                          name="license_number"
                          value={formData.license_number}
                          onChange={handleChange}
                          placeholder="AGRI-CERT-2026-X8"
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white font-mono"
                        />
                      </div>

                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">Years of Experience</label>
                        <input
                          type="number"
                          name="years_experience"
                          value={formData.years_experience}
                          onChange={handleChange}
                          min="1"
                          max="45"
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-[11px] font-semibold text-slate-700 mb-1">Professional Bio & Experience Summary</label>
                      <textarea
                        name="bio"
                        rows="2"
                        value={formData.bio}
                        onChange={handleChange}
                        placeholder="Brief summary of research experience, published papers, or extension work..."
                        className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                      />
                    </div>
                  </div>
                )}

                {/* Customer Profile Fields */}
                {role === 'customer' && (
                  <div className="p-4 bg-amber-50/50 rounded-xl border border-amber-100 space-y-3">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">Customer / Buyer Category</label>
                        <select
                          name="customer_type"
                          value={formData.customer_type}
                          onChange={handleChange}
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                        >
                          <option value="Individual Farmer / Buyer">Individual Farmer / Buyer</option>
                          <option value="Wholesale Produce Buyer">Wholesale Produce Buyer</option>
                          <option value="Agricultural Cooperative">Agricultural Cooperative</option>
                          <option value="Home Gardener">Home Gardener</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-[11px] font-semibold text-slate-700 mb-1">Postal PIN Code</label>
                        <input
                          type="text"
                          name="pincode"
                          value={formData.pincode}
                          onChange={handleChange}
                          placeholder="422001"
                          className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-[11px] font-semibold text-slate-700 mb-1">Shipping / Billing Address</label>
                      <input
                        type="text"
                        name="address"
                        value={formData.address}
                        onChange={handleChange}
                        placeholder="House No., Street, Landmark"
                        className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
                      />
                    </div>
                  </div>
                )}

                {/* Administrator Security Fields */}
                {role === 'admin' && (
                  <div className="p-4 bg-slate-100 rounded-xl border border-slate-300 space-y-3">
                    <div>
                      <label className="block text-[11px] font-bold text-slate-800 mb-1">
                        Admin Master Passkey *
                      </label>
                      <input
                        type="password"
                        name="admin_secret"
                        value={formData.admin_secret}
                        onChange={handleChange}
                        placeholder="Enter master authorization passkey"
                        className="w-full px-3 py-1.5 text-xs border border-slate-400 rounded-lg bg-white font-mono"
                        required
                      />
                      <p className="text-[10px] text-slate-500 mt-1">
                        Platform security verification required to provision administrative rights.
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading}
                className={`w-full py-3 px-4 ${currentPortal.themeClasses.btn} text-sm font-bold rounded-xl shadow-sm transition flex items-center justify-center space-x-2 disabled:opacity-50 mt-6`}
              >
                <span>{loading ? 'Provisioning Account...' : `Register into ${currentPortal.name}`}</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </form>

            <div className="text-center pt-4 border-t border-slate-100 mt-6">
              <p className="text-xs text-slate-600">
                Already registered for this portal?{' '}
                <Link
                  to={`/login?portal=${role}`}
                  className={`font-bold ${currentPortal.themeClasses.accentText} hover:underline`}
                >
                  Log into {currentPortal.name} →
                </Link>
              </p>
            </div>
          </div>

          {/* Right Column: Live Permissions Granted Preview Card */}
          <div className="lg:col-span-5 space-y-5">
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
              <div className="flex items-center space-x-2 text-xs font-bold text-slate-700 uppercase tracking-wider mb-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>Permissions Granted Upon Registration</span>
              </div>
              <p className="text-xs text-slate-500 mb-4">
                Creating an account in the <strong className="text-slate-800">{currentPortal.name}</strong> will grant these authenticated capabilities:
              </p>

              <div className="space-y-3">
                {currentPortal.permissions.map((perm, idx) => (
                  <div key={idx} className="flex items-start space-x-3 p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                    <div className="w-5 h-5 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center shrink-0 mt-0.5 text-xs font-bold">
                      ✓
                    </div>
                    <div>
                      <p className="text-xs font-bold text-slate-800 leading-snug">{perm.name}</p>
                      <p className="text-[11px] text-slate-500 mt-0.5">{perm.desc}</p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-5 p-3.5 rounded-xl bg-slate-50 border border-slate-200/60 text-xs text-slate-600">
                <p className="font-semibold text-slate-800 mb-1 flex items-center space-x-1.5">
                  <ShieldCheck className="w-4 h-4 text-sky-600" />
                  <span>Strict Portal Permission Policy:</span>
                </p>
                <p className="text-[11px] text-slate-500">
                  Accounts created here can only access the authorized features of this role. If you need multi-role access, you may register separate profiles or contact an administrator.
                </p>
              </div>
            </div>

            {/* Quick Portal Switcher Banner */}
            <div className="bg-gradient-to-br from-slate-900 to-slate-800 p-5 rounded-2xl text-white shadow-sm">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1">Looking for another portal?</h4>
              <p className="text-xs text-slate-300 mb-3">
                Switch portals anytime or sign in with an existing account.
              </p>
              <div className="flex flex-wrap gap-2">
                {Object.values(PORTAL_CONFIG).map((p) => (
                  <button
                    key={p.id}
                    onClick={() => handleRoleSelect(p.id)}
                    className={`px-2.5 py-1 text-xs rounded-lg font-semibold transition ${
                      role === p.id
                        ? 'bg-emerald-500 text-white'
                        : 'bg-white/10 hover:bg-white/20 text-slate-200'
                    }`}
                  >
                    {p.name.replace(' Portal', '')}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RegisterPage;
