import React from 'react';
import { useAuth } from '../../context/AuthContext';
import { ShieldCheck, Award, Briefcase, Mail, Phone, BookOpen } from 'lucide-react';

const SpecialistProfile = () => {
  const { user } = useAuth();
  const profile = user?.profile || {};

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-16 h-16 rounded-2xl bg-emerald-100 text-emerald-800 flex items-center justify-center font-bold text-2xl">
              {user?.full_name?.charAt(0) || 'D'}
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-bold text-slate-900">{user?.full_name}</h1>
                {profile.is_verified && (
                  <span className="flex items-center space-x-1 bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded-full text-[10px] font-bold">
                    <ShieldCheck className="w-3.5 h-3.5" />
                    <span>Verified Specialist</span>
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-500 mt-0.5">{profile.qualification || 'Agricultural Specialist'}</p>
              <span className="inline-block mt-1 text-[11px] font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded">
                {profile.specialization || 'Agronomy & Soil Physics'}
              </span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 my-6 p-4 bg-slate-50 rounded-xl border border-slate-100 text-xs">
          <div>
            <span className="text-[10px] text-slate-400 font-semibold uppercase block">Experience</span>
            <span className="font-bold text-slate-800">{profile.years_experience || 10}+ Years in Extension & Research</span>
          </div>
          <div>
            <span className="text-[10px] text-slate-400 font-semibold uppercase block">Affiliated Organization</span>
            <span className="font-bold text-slate-800">{profile.organization || 'Agricultural University'}</span>
          </div>
        </div>

        <div>
          <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">Professional Summary</h4>
          <p className="text-xs text-slate-600 leading-relaxed bg-slate-50 p-3.5 rounded-xl border border-slate-200">
            {profile.bio || 'Consulting farmers on irrigation scheduling, fertigation protocols, soil salinity management, and precision agronomy.'}
          </p>
        </div>

        <div className="mt-6 pt-4 border-t border-slate-100 space-y-2 text-xs text-slate-600">
          <p className="flex items-center space-x-2">
            <Mail className="w-4 h-4 text-emerald-600" />
            <span>{user?.email}</span>
          </p>
          <p className="flex items-center space-x-2">
            <Phone className="w-4 h-4 text-emerald-600" />
            <span>{user?.phone || '+91 94231 88776'}</span>
          </p>
        </div>
      </div>
    </div>
  );
};

export default SpecialistProfile;
