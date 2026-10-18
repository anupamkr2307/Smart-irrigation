import React, { createContext, useContext, useState, useEffect } from 'react';
import api from '../services/api';

const AuthContext = createContext(null);

export const DEMO_ACCOUNTS = [
  { role: 'farmer', email: 'farmer@smartirrigation.org', pass: 'Farmer@123', label: 'Farmer (Ramesh Patel)' },
  { role: 'retailer', email: 'retailer@smartirrigation.org', pass: 'Retailer@123', label: 'Retailer (Suresh Agro)' },
  { role: 'customer', email: 'customer@smartirrigation.org', pass: 'Customer@123', label: 'Customer (Ananya Sharma)' },
  { role: 'specialist', email: 'specialist@smartirrigation.org', pass: 'Specialist@123', label: 'Specialist (Dr. Arvind)' },
  { role: 'admin', email: 'admin@smartirrigation.org', pass: 'Admin@123', label: 'Admin (System Admin)' },
];

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem('agri_user');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [token, setToken] = useState(() => localStorage.getItem('agri_token') || null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (token) {
      api.get('/auth/me')
        .then(res => {
          setUser(res.data.user);
          localStorage.setItem('agri_user', JSON.stringify(res.data.user));
        })
        .catch(() => {
          logout();
        });
    }
  }, [token]);

  const login = async (email, password) => {
    setLoading(true);
    try {
      const res = await api.post('/auth/login', { email, password });
      const { access_token, user: loggedUser } = res.data;
      localStorage.setItem('agri_token', access_token);
      localStorage.setItem('agri_user', JSON.stringify(loggedUser));
      setToken(access_token);
      setUser(loggedUser);
      return { success: true, user: loggedUser };
    } catch (err) {
      return {
        success: false,
        error: err.response?.data?.error || 'Login failed. Please check credentials.'
      };
    } finally {
      setLoading(false);
    }
  };

  const register = async (userData) => {
    setLoading(true);
    try {
      const res = await api.post('/auth/register', userData);
      const { access_token, user: newUser } = res.data;
      localStorage.setItem('agri_token', access_token);
      localStorage.setItem('agri_user', JSON.stringify(newUser));
      setToken(access_token);
      setUser(newUser);
      return { success: true, user: newUser };
    } catch (err) {
      return {
        success: false,
        error: err.response?.data?.error || 'Registration failed.'
      };
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    try {
      api.post('/auth/logout').catch(() => {});
    } catch {}
    localStorage.removeItem('agri_token');
    localStorage.removeItem('agri_user');
    setToken(null);
    setUser(null);
  };

  const quickLoginAs = async (roleName) => {
    const acc = DEMO_ACCOUNTS.find(a => a.role === roleName);
    if (acc) {
      return await login(acc.email, acc.pass);
    }
  };

  const value = {
    user,
    token,
    loading,
    login,
    register,
    logout,
    quickLoginAs,
    isAuthenticated: !!user,
    isFarmer: user?.role === 'farmer',
    isRetailer: user?.role === 'retailer',
    isCustomer: user?.role === 'customer',
    isSpecialist: user?.role === 'specialist',
    isAdmin: user?.role === 'admin',
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => useContext(AuthContext);
