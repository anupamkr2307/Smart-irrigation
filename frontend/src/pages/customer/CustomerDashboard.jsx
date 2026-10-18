import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useUnit } from '../../context/UnitContext';
import { ShoppingCart, Search, Filter, CheckCircle2, Store } from 'lucide-react';
import toast from 'react-hot-toast';

const CustomerDashboard = () => {
  const { formatCurrency } = useUnit();
  const [products, setProducts] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedCat, setSelectedCat] = useState('');
  const [search, setSearch] = useState('');
  const [cart, setCart] = useState(null);
  const [loading, setLoading] = useState(true);

  // Checkout modal
  const [showCheckout, setShowCheckout] = useState(false);
  const [shipping, setShipping] = useState({
    name: 'Ananya Sharma',
    phone: '+91 97654 32109',
    address: 'Green Meadows, Villa 12',
    city: 'Nashik',
    pincode: '422003'
  });

  useEffect(() => {
    loadData();
  }, [selectedCat, search]);

  const loadData = async () => {
    setLoading(true);
    try {
      let url = '/marketplace/products?';
      if (selectedCat) url += `category=${selectedCat}&`;
      if (search) url += `search=${encodeURIComponent(search)}&`;

      const [prodRes, catRes, cartRes] = await Promise.all([
        api.get(url),
        api.get('/marketplace/categories'),
        api.get('/marketplace/cart').catch(() => ({ data: { cart: null } }))
      ]);

      setProducts(prodRes.data.products || []);
      setCategories(catRes.data.categories || []);
      setCart(cartRes.data.cart || null);
    } catch (err) {
      console.error(err);
      toast.error('Failed to load marketplace');
    } finally {
      setLoading(false);
    }
  };

  const handleAddToCart = async (product) => {
    try {
      const res = await api.post('/marketplace/cart/items', {
        product_id: product.id,
        quantity: 1
      });
      setCart(res.data.cart);
      toast.success(`Added ${product.name} to cart!`);
    } catch (err) {
      toast.error(err.response?.data?.error || 'Failed to add item');
    }
  };

  const handleCheckout = async (e) => {
    e.preventDefault();
    try {
      await api.post('/marketplace/checkout', {
        shipping,
        payment_method: 'Mock Card / NetBanking (Test Mode)'
      });
      toast.success('Payment verified! Your order has been placed.');
      setShowCheckout(false);
      setCart(null);
      loadData();
    } catch (err) {
      toast.error(err.response?.data?.error || 'Checkout failed');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest flex items-center space-x-1">
            <Store className="w-3.5 h-3.5" />
            <span>Agricultural Supplies Marketplace</span>
          </span>
          <h1 className="text-2xl font-extrabold text-slate-900 mt-1">Shop Farm Equipment & Inputs</h1>
          <p className="text-xs text-slate-500 mt-1">Direct from certified regional agricultural retailers with verified delivery.</p>
        </div>

        <button
          onClick={() => setShowCheckout(true)}
          className="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold shadow-xs transition flex items-center space-x-2"
        >
          <ShoppingCart className="w-4 h-4" />
          <span>Cart ({cart?.total_items || 0} items • {formatCurrency(cart?.total_amount || 0)})</span>
        </button>
      </div>

      {/* Filters & Search */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search drip kits, seeds, organic fertilizers, soil sensors..."
            className="w-full pl-9 pr-4 py-2.5 text-xs border border-slate-300 rounded-xl bg-white outline-none focus:ring-2 focus:ring-emerald-500"
          />
        </div>

        <select
          value={selectedCat}
          onChange={(e) => setSelectedCat(e.target.value)}
          className="px-4 py-2.5 text-xs border border-slate-300 rounded-xl bg-white font-medium"
        >
          <option value="">All Categories</option>
          {categories.map((c) => (
            <option key={c.id} value={c.slug}>{c.name}</option>
          ))}
        </select>
      </div>

      {/* Product Catalog Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {products.map((p) => (
          <div key={p.id} className="bg-white rounded-2xl overflow-hidden border border-slate-200 shadow-xs flex flex-col justify-between">
            {p.image_url && (
              <div className="h-48 w-full bg-slate-100 overflow-hidden">
                <img src={p.image_url} alt={p.name} className="w-full h-full object-cover" />
              </div>
            )}
            <div className="p-5 flex-1 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded bg-emerald-50 text-emerald-800">
                  {p.category_name}
                </span>
                <h4 className="text-sm font-bold text-slate-900 mt-2">{p.name}</h4>
                <p className="text-xs text-slate-500 line-clamp-2 mt-1 leading-relaxed">{p.description}</p>
                <span className="text-[11px] text-slate-400 block mt-2">
                  Seller: <strong>{p.shop_name || p.retailer_name}</strong>
                </span>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between">
                <div>
                  <span className="text-base font-extrabold text-slate-900 block">{formatCurrency(p.price)}</span>
                  <span className="text-[10px] text-slate-400">per {p.unit}</span>
                </div>

                <button
                  onClick={() => handleAddToCart(p)}
                  disabled={p.stock_quantity <= 0}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-semibold shadow-xs transition disabled:opacity-50"
                >
                  {p.stock_quantity > 0 ? 'Add to Cart' : 'Out of Stock'}
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Checkout Modal */}
      {showCheckout && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl p-6 max-w-lg w-full space-y-4 shadow-xl border border-slate-200 animate-fade-in">
            <h3 className="text-base font-bold text-slate-900 pb-2 border-b border-slate-100">
              Checkout & Mock Payment
            </h3>

            {!cart || cart.items?.length === 0 ? (
              <p className="text-xs text-slate-500 py-6 text-center">Your shopping cart is empty.</p>
            ) : (
              <div className="space-y-3">
                <div className="max-h-36 overflow-y-auto space-y-2">
                  {cart.items.map((i) => (
                    <div key={i.id} className="flex justify-between text-xs py-1 px-2.5 bg-slate-50 rounded">
                      <span>{i.product_name} × {i.quantity}</span>
                      <span className="font-bold">{formatCurrency(i.subtotal)}</span>
                    </div>
                  ))}
                </div>

                <form onSubmit={handleCheckout} className="space-y-3 pt-2 border-t border-slate-100">
                  <div>
                    <label className="block text-[11px] font-semibold text-slate-700 mb-1">Full Name</label>
                    <input
                      type="text"
                      required
                      value={shipping.name}
                      onChange={(e) => setShipping({ ...shipping, name: e.target.value })}
                      className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-700 mb-1">Phone</label>
                      <input
                        type="text"
                        required
                        value={shipping.phone}
                        onChange={(e) => setShipping({ ...shipping, phone: e.target.value })}
                        className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg"
                      />
                    </div>
                    <div>
                      <label className="block text-[11px] font-semibold text-slate-700 mb-1">PIN Code</label>
                      <input
                        type="text"
                        required
                        value={shipping.pincode}
                        onChange={(e) => setShipping({ ...shipping, pincode: e.target.value })}
                        className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-[11px] font-semibold text-slate-700 mb-1">Address & City</label>
                    <input
                      type="text"
                      required
                      value={shipping.address}
                      onChange={(e) => setShipping({ ...shipping, address: e.target.value })}
                      className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg"
                    />
                  </div>

                  <div className="p-3 bg-emerald-50 rounded-xl border border-emerald-200 text-xs flex justify-between items-center">
                    <span>Total Amount Payable:</span>
                    <strong className="text-base text-emerald-800">{formatCurrency(cart.total_amount)}</strong>
                  </div>

                  <div className="flex justify-end space-x-2 pt-2">
                    <button
                      type="button"
                      onClick={() => setShowCheckout(false)}
                      className="px-3 py-1.5 text-xs text-slate-600"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-bold"
                    >
                      Pay & Place Order (Mock Test Mode)
                    </button>
                  </div>
                </form>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default CustomerDashboard;
