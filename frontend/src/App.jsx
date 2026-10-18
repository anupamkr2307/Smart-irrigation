import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { UnitProvider } from './context/UnitContext';
import { Toaster } from 'react-hot-toast';

// Common Components
import QuickDemoBar from './components/QuickDemoBar';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Footer from './components/Footer';

// Public Pages
import LandingPage from './pages/public/LandingPage';
import AboutPage from './pages/public/AboutPage';
import FeaturesPage from './pages/public/FeaturesPage';
import LoginPage from './pages/public/LoginPage';
import RegisterPage from './pages/public/RegisterPage';
import ContactPage from './pages/public/ContactPage';

// Farmer Pages
import FarmerDashboard from './pages/farmer/FarmerDashboard';
import FarmManager from './pages/farmer/FarmManager';
import IrrigationCalculator from './pages/farmer/IrrigationCalculator';
import CropRecommender from './pages/farmer/CropRecommender';
import FarmerAdvisories from './pages/farmer/FarmerAdvisories';
import ConsultationCenter from './pages/farmer/ConsultationCenter';
import FarmerMarketplace from './pages/farmer/FarmerMarketplace';

// Retailer Pages
import RetailerDashboard from './pages/retailer/RetailerDashboard';
import ProductInventory from './pages/retailer/ProductInventory';
import RetailerOrders from './pages/retailer/RetailerOrders';
import QuotationDesk from './pages/retailer/QuotationDesk';

// Customer Pages
import CustomerDashboard from './pages/customer/CustomerDashboard';
import CustomerOrders from './pages/customer/CustomerOrders';

// Specialist Pages
import SpecialistDashboard from './pages/specialist/SpecialistDashboard';
import PublishAdvisory from './pages/specialist/PublishAdvisory';
import SpecialistProfile from './pages/specialist/SpecialistProfile';

// Admin Pages
import AdminDashboard from './pages/admin/AdminDashboard';
import UserManagement from './pages/admin/UserManagement';
import MLAnalytics from './pages/admin/MLAnalytics';
import AuditLogs from './pages/admin/AuditLogs';

// Protected Route Component
const ProtectedRoute = ({ allowedRoles }) => {
  const { user, isAuthenticated, loading } = useAuth();

  if (loading) {
    return <div className="p-10 text-center text-xs text-slate-500">Loading user session...</div>;
  }

  if (!isAuthenticated || !user) {
    return <Navigate to="/login" replace />;
  }

  if (allowedRoles && !allowedRoles.includes(user.role) && user.role !== 'admin') {
    return (
      <div className="p-10 text-center space-y-2">
        <h2 className="text-lg font-bold text-slate-900">Access Restricted</h2>
        <p className="text-xs text-slate-500">Your role ({user.role}) is not authorized to access this section.</p>
        <Navigate to={`/${user.role}/dashboard`} replace />
      </div>
    );
  }

  return (
    <div className="flex">
      <Sidebar />
      <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto max-w-7xl">
        <Outlet />
      </main>
    </div>
  );
};

// Public Layout
const PublicLayout = () => {
  return (
    <div className="flex flex-col min-h-screen">
      <Outlet />
      <Footer />
    </div>
  );
};

function App() {
  return (
    <AuthProvider>
      <UnitProvider>
        <BrowserRouter>
          <Toaster position="top-right" />
          <QuickDemoBar />
          <Navbar />

          <Routes>
            {/* Public Pages */}
            <Route element={<PublicLayout />}>
              <Route path="/" element={<LandingPage />} />
              <Route path="/about" element={<AboutPage />} />
              <Route path="/features" element={<FeaturesPage />} />
              <Route path="/marketplace" element={<CustomerDashboard />} />
              <Route path="/login" element={<LoginPage />} />
              <Route path="/register" element={<RegisterPage />} />
              <Route path="/contact" element={<ContactPage />} />
            </Route>

            {/* Farmer Routes */}
            <Route element={<ProtectedRoute allowedRoles={['farmer']} />}>
              <Route path="/farmer/dashboard" element={<FarmerDashboard />} />
              <Route path="/farmer/farms" element={<FarmManager />} />
              <Route path="/farmer/irrigation" element={<IrrigationCalculator />} />
              <Route path="/farmer/crops" element={<CropRecommender />} />
              <Route path="/farmer/advisories" element={<FarmerAdvisories />} />
              <Route path="/farmer/consultations" element={<ConsultationCenter />} />
              <Route path="/farmer/marketplace" element={<FarmerMarketplace />} />
            </Route>

            {/* Retailer Routes */}
            <Route element={<ProtectedRoute allowedRoles={['retailer']} />}>
              <Route path="/retailer/dashboard" element={<RetailerDashboard />} />
              <Route path="/retailer/inventory" element={<ProductInventory />} />
              <Route path="/retailer/orders" element={<RetailerOrders />} />
              <Route path="/retailer/quotations" element={<QuotationDesk />} />
            </Route>

            {/* Customer Routes */}
            <Route element={<ProtectedRoute allowedRoles={['customer']} />}>
              <Route path="/customer/dashboard" element={<CustomerDashboard />} />
              <Route path="/customer/orders" element={<CustomerOrders />} />
            </Route>

            {/* Specialist Routes */}
            <Route element={<ProtectedRoute allowedRoles={['specialist']} />}>
              <Route path="/specialist/dashboard" element={<SpecialistDashboard />} />
              <Route path="/specialist/publish-advisory" element={<PublishAdvisory />} />
              <Route path="/specialist/profile" element={<SpecialistProfile />} />
            </Route>

            {/* Admin Routes */}
            <Route element={<ProtectedRoute allowedRoles={['admin']} />}>
              <Route path="/admin/dashboard" element={<AdminDashboard />} />
              <Route path="/admin/users" element={<UserManagement />} />
              <Route path="/admin/ml-metrics" element={<MLAnalytics />} />
              <Route path="/admin/audit-logs" element={<AuditLogs />} />
            </Route>

            {/* Catch-all redirect */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </UnitProvider>
    </AuthProvider>
  );
}

export default App;
