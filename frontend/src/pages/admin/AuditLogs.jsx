import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { Shield, Clock, Search } from 'lucide-react';
import toast from 'react-hot-toast';

const AuditLogs = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');

  useEffect(() => {
    loadLogs();
  }, []);

  const loadLogs = async () => {
    setLoading(true);
    try {
      const res = await api.get('/admin/audit-logs');
      setLogs(res.data.audit_logs || []);
    } catch (err) {
      console.error(err);
      toast.error('Failed to load audit logs');
    } finally {
      setLoading(false);
    }
  };

  const filteredLogs = logs.filter((l) => {
    if (!search) return true;
    const term = search.toLowerCase();
    return (
      l.action?.toLowerCase().includes(term) ||
      l.user_email?.toLowerCase().includes(term) ||
      l.details?.toLowerCase().includes(term)
    );
  });

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900">Security & Operational Audit Logs</h1>
          <p className="text-xs text-slate-500 mt-1">Immutable trail tracking logins, permissions, checkouts, and model evaluations.</p>
        </div>

        <div className="relative w-full sm:w-64">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search action or user..."
            className="w-full pl-9 pr-3 py-1.5 text-xs border border-slate-300 rounded-lg outline-none"
          />
        </div>
      </div>

      <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-500 uppercase font-semibold">
              <tr>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4">Action</th>
                <th className="py-3 px-4">User</th>
                <th className="py-3 px-4">Resource</th>
                <th className="py-3 px-4">IP Address</th>
                <th className="py-3 px-4">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {filteredLogs.length === 0 ? (
                <tr>
                  <td colSpan="6" className="py-8 text-center text-slate-400">
                    No matching audit records found.
                  </td>
                </tr>
              ) : (
                filteredLogs.map((log) => (
                  <tr key={log.id} className="hover:bg-slate-50/70 transition">
                    <td className="py-3 px-4 text-slate-500 font-mono text-[11px]">
                      {log.created_at ? new Date(log.created_at).toLocaleString() : 'N/A'}
                    </td>
                    <td className="py-3 px-4 font-bold text-slate-900">
                      <span className="bg-slate-100 text-slate-800 px-2 py-0.5 rounded text-[10px] font-mono">
                        {log.action}
                      </span>
                    </td>
                    <td className="py-3 px-4">{log.user_email}</td>
                    <td className="py-3 px-4 font-mono text-[11px] text-slate-500">{log.resource || 'System'}</td>
                    <td className="py-3 px-4 font-mono text-[11px] text-slate-400">{log.ip_address || '127.0.0.1'}</td>
                    <td className="py-3 px-4 text-xs text-slate-600 max-w-xs truncate">{log.details}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default AuditLogs;
