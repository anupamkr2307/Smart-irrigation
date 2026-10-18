import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { Users, ShieldCheck, Check, X, Filter } from 'lucide-react';
import toast from 'react-hot-toast';

const UserManagement = () => {
  const [users, setUsers] = useState([]);
  const [roleFilter, setRoleFilter] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadUsers();
  }, [roleFilter]);

  const loadUsers = async () => {
    setLoading(true);
    try {
      const url = roleFilter ? `/admin/users?role=${roleFilter}` : '/admin/users';
      const res = await api.get(url);
      setUsers(res.data.users || []);
    } catch (err) {
      console.error(err);
      toast.error('Failed to load users');
    } finally {
      setLoading(false);
    }
  };

  const handleToggleStatus = async (user) => {
    try {
      await api.patch(`/admin/users/${user.id}/status`, { is_active: !user.is_active });
      toast.success(`User ${user.email} status updated`);
      loadUsers();
    } catch (err) {
      toast.error(err.response?.data?.error || 'Action failed');
    }
  };

  const handleVerifySpecialist = async (user) => {
    const isCurrentlyVerified = user.profile?.is_verified;
    try {
      await api.post(`/admin/specialists/verify/${user.id}`, { is_verified: !isCurrentlyVerified });
      toast.success(`Specialist verification status set to ${!isCurrentlyVerified}`);
      loadUsers();
    } catch (err) {
      toast.error(err.response?.data?.error || 'Action failed');
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900">User & Account Governance</h1>
          <p className="text-xs text-slate-500 mt-1">Audit platform accounts, toggle access status, and verify agricultural specialists.</p>
        </div>

        <div className="flex items-center space-x-2">
          <Filter className="w-4 h-4 text-slate-400" />
          <select
            value={roleFilter}
            onChange={(e) => setRoleFilter(e.target.value)}
            className="px-3 py-1.5 text-xs border border-slate-300 rounded-lg bg-white"
          >
            <option value="">All Roles</option>
            <option value="farmer">Farmer</option>
            <option value="retailer">Retailer</option>
            <option value="specialist">Specialist</option>
            <option value="customer">Customer</option>
            <option value="admin">Administrator</option>
          </select>
        </div>
      </div>

      <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-500 uppercase font-semibold">
              <tr>
                <th className="py-3 px-4">User</th>
                <th className="py-3 px-4">Role</th>
                <th className="py-3 px-4">Profile Info</th>
                <th className="py-3 px-4">Account Status</th>
                <th className="py-3 px-4">Verification</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {users.map((u) => (
                <tr key={u.id} className="hover:bg-slate-50/70 transition">
                  <td className="py-3.5 px-4">
                    <strong className="text-slate-900 block font-bold">{u.full_name}</strong>
                    <span className="text-[11px] text-slate-500">{u.email}</span>
                  </td>
                  <td className="py-3.5 px-4">
                    <span className="capitalize font-semibold px-2 py-0.5 rounded-full text-[10px] bg-slate-100 text-slate-800">
                      {u.role}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 text-[11px] text-slate-600">
                    {u.role === 'farmer' && (u.profile?.farm_name || `${u.profile?.district || 'India'}`)}
                    {u.role === 'retailer' && (u.profile?.shop_name || 'Agro Shop')}
                    {u.role === 'specialist' && (u.profile?.specialization || 'Agronomist')}
                    {u.role === 'customer' && 'General Customer'}
                    {u.role === 'admin' && 'System Administrator'}
                  </td>
                  <td className="py-3.5 px-4">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase ${
                      u.is_active ? 'bg-emerald-100 text-emerald-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {u.is_active ? 'Active' : 'Deactivated'}
                    </span>
                  </td>
                  <td className="py-3.5 px-4">
                    {u.role === 'specialist' ? (
                      <button
                        onClick={() => handleVerifySpecialist(u)}
                        className={`px-2 py-1 rounded text-[10px] font-bold flex items-center space-x-1 ${
                          u.profile?.is_verified
                            ? 'bg-emerald-100 text-emerald-800 hover:bg-emerald-200'
                            : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                        }`}
                      >
                        <ShieldCheck className="w-3.5 h-3.5" />
                        <span>{u.profile?.is_verified ? 'Verified (Toggle)' : 'Pending (Verify)'}</span>
                      </button>
                    ) : (
                      <span className="text-[11px] text-slate-400">N/A</span>
                    )}
                  </td>
                  <td className="py-3.5 px-4 text-right">
                    {u.role !== 'admin' && (
                      <button
                        onClick={() => handleToggleStatus(u)}
                        className={`px-2.5 py-1 rounded text-[10px] font-bold ${
                          u.is_active
                            ? 'bg-red-50 text-red-700 hover:bg-red-100'
                            : 'bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
                        }`}
                      >
                        {u.is_active ? 'Deactivate' : 'Activate'}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default UserManagement;
