import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useUnit } from '../../context/UnitContext';
import { ShoppingBag, Plus, ShoppingCart, Tag, Send, CheckCircle2, Store, Truck } from 'lucide-react';
import toast from 'react-hot-toast';

const FarmerMarketplace = () => {
  const { formatCurrency } = useUnit();
  const [products, setProducts] = useState([]);
  const [cart, setCart] = useState(null);
  const [quotes, setQuotes] = useState([]);
  const [activeTab, setActiveTab] = useState('browse'); // 'browse' | 'quotes' | 'cart'
  const [loading, setLoading] = useState(true);

  // New quote form
  const [itemTitle, setItemTitle] = useState('');
  const [quantity, setQuantity] = useState('500 meters');
  const [farmLocation, setFarmLocation] = useState('Nashik, Maharashtra');
  const [quoteDesc, setQuoteDesc] = useState('');
  const [showQuoteModal, setShowQuoteModal] = useState(false);

  // Checkout form
  const [shipping, setShipping] = useState({
    name: 'Ramesh Patel',
    phone: '+91 98765 43210',
    address: 'Patel Farmhouse, Market Road',
    city: 'Nashik',
    pincode: '422003'
  });

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [prodRes, cartRes, quoteRes] = await Promise.all([
        api.get('/marketplace/products'),
        api.get('/marketplace/cart').catch(() => ({ data: { cart: null } })),
        api.get('/marketplace/quotations').catch(() => ({ data: { quotation_requests: [] } }))
      ]);
      setProducts(prodRes.data.products || []);
      setCart(cartRes.data.cart || null);
      setQuotes(quoteRes.data.quotation_requests || []);
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

  const handleRemoveItem = async (itemId) => {
    try {
      const res = await api.delete(`/marketplace/cart/items/${itemId}`);
      setCart(res.data.cart);
      toast.success('Item removed');
    } catch (err) {
      toast.error('Failed to remove item');
    }
  };

  const handleCheckout = async (e) => {
    e.preventDefault();
    try {
      const res = await api.post('/marketplace/checkout', {
        shipping,
        payment_method: 'Mock Farmer UPI / Direct Agri Credit'
      });
      toast.success('Order placed successfully! (Mock Payment Approved)');
      setCart(null);
      setActiveTab('browse');
      loadData();
    } catch (err) {
      toast.error(err.response?.data?.error || 'Checkout failed');
    }
  };

  const handleCreateQuote = async (e) => {
    e.preventDefault();
    try {
      await api.post('/marketplace/quotations', {
        item_title: itemTitle,
        quantity,
        farm_location: farmLocation,
        description: quoteDesc
      });
      toast.success('Quotation request broadcast to retailers!');
      setShowQuoteModal(false);
      setItemTitle('');
      setQuoteDesc('');
      loadData();
    } catch (err) {
      toast.error('Failed to request quotation');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest flex items-center space-x-1">
            <Store className="w-3.5 h-3.5" />
            <span>Agricultural Inputs & Equipment</span>
          </span>
          <h1 className="text-2xl font-extrabold text-slate-900 mt-1">Farmer Marketplace & Quotations</h1>
          <p className="text-xs text-slate-500 mt-1">
            Procure certified irrigation kits, moisture meters, seeds, and request bulk price quotations directly from retailers.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowQuoteModal(true)}
            className="px-3.5 py-2 bg-emerald-50 hover:bg-emerald-100 text-emerald-800 rounded-xl text-xs font-semibold border border-emerald-200 transition flex items-center space-x-1"
          >
            <Tag className="w-4 h-4" />
            <span>Request Bulk Quotation</span>
          </button>

          <button
            onClick={() => setActiveTab(activeTab === 'cart' ? 'browse' : 'cart')}
            className="px-3.5 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold shadow-xs transition flex items-center space-x-1.5"
          >
            <ShoppingCart className="w-4 h-4" />
            <span>Cart ({cart?.total_items || 0})</span>
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-200 space-x-4">
        <button
          onClick={() => setActiveTab('browse')}
          className={`pb-3 text-xs font-bold transition ${
            activeTab === 'browse'
              ? 'border-b-2 border-emerald-600 text-emerald-700'
              : 'text-slate-500 hover:text-slate-800'
          }`}
        >
          Product Catalog ({products.length})
        </button>
        <button
          onClick={() => setActiveTab('quotes')}
          className={`pb-3 text-xs font-bold transition ${
            activeTab === 'quotes'
              ? 'border-b-2 border-emerald-600 text-emerald-700'
              : 'text-slate-500 hover:text-slate-800'
          }`}
        >
          My Quotation Requests ({quotes.length})
        </button>
        <button
          onClick={() => setActiveTab('cart')}
          className={`pb-3 text-xs font-bold transition ${
            activeTab === 'cart'
              ? 'border-b-2 border-emerald-600 text-emerald-700'
              : 'text-slate-500 hover:text-slate-800'
          }`}
        >
          Shopping Cart ({cart?.total_items || 0})
        </button>
      </div>

      {/* Browse Products */}
      {activeTab === 'browse' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {products.map((p) => (
            <div key={p.id} className="bg-white rounded-2xl overflow-hidden border border-slate-200 shadow-xs flex flex-col justify-between">
              {p.image_url && (
                <div className="h-44 w-full bg-slate-100 overflow-hidden">
                  <img src={p.image_url} alt={p.name} className="w-full h-full object-cover" />
                </div>
              )}
              <div className="p-5 flex-1 flex flex-col justify-between">
                <div>
                  <div className="flex justify-between items-start mb-1">
                    <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded bg-emerald-50 text-emerald-800">
                      {p.category_name}
                    </span>
                    <span className="text-xs font-semibold text-slate-500">
                      Stock: <strong>{p.stock_quantity}</strong>
                    </span>
                  </div>
                  <h4 className="text-sm font-bold text-slate-900 mt-2">{p.name}</h4>
                  <p className="text-xs text-slate-500 line-clamp-2 mt-1 leading-relaxed">{p.description}</p>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between">
                  <div>
                    <span className="text-base font-extrabold text-slate-900 block">
                      {formatCurrency(p.price)}
                    </span>
                    <span className="text-[10px] text-slate-400">per {p.unit}</span>
                  </div>

                  <button
                    onClick={() => handleAddToCart(p)}
                    disabled={p.stock_quantity <= 0}
                    className="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-semibold shadow-xs transition disabled:opacity-50"
                  >
                    {p.stock_quantity > 0 ? 'Add to Cart' : 'Out of Stock'}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Quotation Requests Tab */}
      {activeTab === 'quotes' && (
        <div className="space-y-4">
          {quotes.length === 0 ? (
            <div className="bg-white rounded-2xl p-10 border border-slate-200 text-center text-xs text-slate-500">
              No quotation requests posted yet. Click "Request Bulk Quotation" above to receive custom prices from retailers.
            </div>
          ) : (
            quotes.map((q) => (
              <div key={q.id} className="bg-white rounded-2xl p-5 border border-slate-200 shadow-xs space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="text-sm font-bold text-slate-900">{q.item_title}</h4>
                    <span className="text-xs text-slate-500">Quantity: <strong>{q.quantity}</strong> • Location: {q.farm_location}</span>
                  </div>
                  <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded bg-sky-100 text-sky-800">
                    Status: {q.status}
                  </span>
                </div>
                <p className="text-xs text-slate-600">{q.description}</p>

                {/* Quotations received from retailers */}
                {q.quotations && q.quotations.length > 0 ? (
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
                    <h5 className="text-[11px] font-bold text-slate-700 uppercase">Offers Received from Retailers:</h5>
                    {q.quotations.map((offer) => (
                      <div key={offer.id} className="flex justify-between items-center bg-white p-2.5 rounded-lg border border-slate-200">
                        <div>
                          <strong className="text-xs text-slate-900">{offer.retailer_name} ({offer.shop_name || 'Retailer'})</strong>
                          <p className="text-[11px] text-slate-500">Est. Delivery: {offer.estimated_delivery_days} days • {offer.remarks}</p>
                        </div>
                        <div className="text-right">
                          <span className="text-sm font-bold text-emerald-700 block">{formatCurrency(offer.offered_price)}</span>
                          <span className="text-[10px] text-slate-400">Total Quote</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-[11px] text-slate-400 italic">Waiting for retailer bids...</p>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {/* Cart & Checkout Tab */}
      {activeTab === 'cart' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7 bg-white rounded-2xl p-6 border border-slate-200 shadow-xs space-y-4">
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider pb-2 border-b border-slate-100">
              Selected Equipment & Inputs
            </h3>

            {!cart || cart.items?.length === 0 ? (
              <p className="text-xs text-slate-500 py-6 text-center">Your shopping cart is empty.</p>
            ) : (
              <div className="space-y-3">
                {cart.items.map((item) => (
                  <div key={item.id} className="flex justify-between items-center p-3 bg-slate-50 rounded-xl border border-slate-200">
                    <div>
                      <h4 className="text-xs font-bold text-slate-900">{item.product_name}</h4>
                      <span className="text-[11px] text-slate-500">
                        Qty: {item.quantity} × {formatCurrency(item.price)}
                      </span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <span className="text-xs font-bold text-slate-800">{formatCurrency(item.subtotal)}</span>
                      <button
                        onClick={() => handleRemoveItem(item.id)}
                        className="text-xs text-red-500 hover:text-red-700"
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="lg:col-span-5 bg-white rounded-2xl p-6 border border-slate-200 shadow-xs space-y-4">
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider pb-2 border-b border-slate-100">
              Delivery & Mock Checkout
            </h3>

            <form onSubmit={handleCheckout} className="space-y-3">
              <div>
                <label className="block text-[11px] font-semibold text-slate-700 mb-1">Recipient Name</label>
                <input
                  type="text"
                  required
                  value={shipping.name}
                  onChange={(e) => setShipping({ ...shipping, name: e.target.value })}
                  className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div>
                <label className="block text-[11px] font-semibold text-slate-700 mb-1">Contact Phone</label>
                <input
                  type="text"
                  required
                  value={shipping.phone}
                  onChange={(e) => setShipping({ ...shipping, phone: e.target.value })}
                  className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div>
                <label className="block text-[11px] font-semibold text-slate-700 mb-1">Farm / Delivery Address</label>
                <textarea
                  rows={2}
                  required
                  value={shipping.address}
                  onChange={(e) => setShipping({ ...shipping, address: e.target.value })}
                  className="w-full px-3 py-1.5 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[11px] font-semibold text-slate-700 mb-1">City / District</label>
                  <input
                    type="text"
                    required
                    value={shipping.city}
                    onChange={(e) => setShipping({ ...shipping, city: e.target.value })}
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

              <div className="pt-3 border-t border-slate-100 flex justify-between items-center">
                <span className="text-xs text-slate-500">Order Total:</span>
                <span className="text-base font-extrabold text-slate-900">{formatCurrency(cart?.total_amount || 0)}</span>
              </div>

              <button
                type="submit"
                disabled={!cart || cart.items?.length === 0}
                className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold shadow-xs transition disabled:opacity-50"
              >
                Confirm Order (Mock Payment)
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Bulk Quote Modal */}
      {showQuoteModal && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl p-6 max-w-lg w-full space-y-4 shadow-xl border border-slate-200">
            <h3 className="text-base font-bold text-slate-900 pb-2 border-b border-slate-100">
              Request Bulk Quotation from Retailers
            </h3>

            <form onSubmit={handleCreateQuote} className="space-y-3">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Equipment / Input Name *</label>
                <input
                  type="text"
                  required
                  value={itemTitle}
                  onChange={(e) => setItemTitle(e.target.value)}
                  placeholder="e.g. 16mm Inline Drip Laterals (Class 2)"
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Quantity Needed *</label>
                  <input
                    type="text"
                    required
                    value={quantity}
                    onChange={(e) => setQuantity(e.target.value)}
                    placeholder="e.g. 1,000 meters"
                    className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Delivery Location *</label>
                  <input
                    type="text"
                    required
                    value={farmLocation}
                    onChange={(e) => setFarmLocation(e.target.value)}
                    placeholder="e.g. Nashik, Maharashtra"
                    className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Technical Specifications / Notes</label>
                <textarea
                  rows={3}
                  value={quoteDesc}
                  onChange={(e) => setQuoteDesc(e.target.value)}
                  placeholder="Mention emitter spacing (40cm), discharge rate (2 LPH), or brand preference..."
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div className="flex justify-end space-x-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowQuoteModal(false)}
                  className="px-4 py-2 border border-slate-300 text-slate-600 rounded-lg text-xs font-semibold"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-xs font-bold"
                >
                  Broadcast Request
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default FarmerMarketplace;
