import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { useUnit } from '../../context/UnitContext';
import { Package, Plus, Edit, Trash2, CheckCircle, Tag } from 'lucide-react';
import toast from 'react-hot-toast';

const ProductInventory = () => {
  const { user } = useAuth();
  const { formatCurrency } = useUnit();
  const [products, setProducts] = useState([]);
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddModal, setShowAddModal] = useState(false);

  // New product state
  const [name, setName] = useState('');
  const [categoryId, setCategoryId] = useState(1);
  const [price, setPrice] = useState(1200);
  const [unit, setUnit] = useState('piece');
  const [stock, setStock] = useState(25);
  const [description, setDescription] = useState('');
  const [imageUrl, setImageUrl] = useState('');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [prodRes, catRes] = await Promise.all([
        api.get(`/marketplace/products?retailer_id=${user.id}`),
        api.get('/marketplace/categories')
      ]);
      setProducts(prodRes.data.products || []);
      setCategories(catRes.data.categories || []);
      if (catRes.data.categories?.length > 0) {
        setCategoryId(catRes.data.categories[0].id);
      }
    } catch (err) {
      console.error(err);
      toast.error('Failed to load inventory');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateProduct = async (e) => {
    e.preventDefault();
    if (!name || price <= 0) {
      toast.error('Product name and valid price are required.');
      return;
    }

    try {
      await api.post('/marketplace/products', {
        name,
        category_id: parseInt(categoryId),
        price: parseFloat(price),
        unit,
        stock_quantity: parseInt(stock),
        description,
        image_url: imageUrl || 'https://images.unsplash.com/photo-1592417817098-8f3d6eb228cc?auto=format&fit=crop&w=600&q=80'
      });
      toast.success('Product listed in marketplace!');
      setShowAddModal(false);
      setName('');
      setDescription('');
      setImageUrl('');
      loadData();
    } catch (err) {
      toast.error('Failed to add product');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-xs flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <span className="text-xs font-bold text-sky-600 uppercase tracking-widest flex items-center space-x-1">
            <Package className="w-3.5 h-3.5" />
            <span>Store Inventory Management</span>
          </span>
          <h1 className="text-2xl font-extrabold text-slate-900 mt-1">Product Catalog & Stock</h1>
          <p className="text-xs text-slate-500 mt-1">
            Maintain your store's irrigation hardware, high-yield seeds, fertilizers, and precision monitoring tools.
          </p>
        </div>

        <button
          onClick={() => setShowAddModal(true)}
          className="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold shadow-xs transition flex items-center space-x-1.5"
        >
          <Plus className="w-4 h-4" />
          <span>Add New Product</span>
        </button>
      </div>

      {/* Inventory Table */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-500 uppercase font-semibold">
              <tr>
                <th className="py-3 px-4">Product Name</th>
                <th className="py-3 px-4">Category</th>
                <th className="py-3 px-4">Price</th>
                <th className="py-3 px-4">Stock Level</th>
                <th className="py-3 px-4">Rating</th>
                <th className="py-3 px-4">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {products.length === 0 ? (
                <tr>
                  <td colSpan="6" className="py-8 text-center text-slate-400">
                    No products listed. Click "Add New Product" to populate your catalog.
                  </td>
                </tr>
              ) : (
                products.map((p) => (
                  <tr key={p.id} className="hover:bg-slate-50/70 transition">
                    <td className="py-3.5 px-4 font-bold text-slate-900">
                      <div className="flex items-center space-x-2">
                        {p.image_url && (
                          <img src={p.image_url} alt={p.name} className="w-8 h-8 rounded object-cover" />
                        )}
                        <span>{p.name}</span>
                      </div>
                    </td>
                    <td className="py-3.5 px-4">{p.category_name}</td>
                    <td className="py-3.5 px-4 font-bold text-slate-900">
                      {formatCurrency(p.price)} <span className="text-[10px] text-slate-400 font-normal">/ {p.unit}</span>
                    </td>
                    <td className="py-3.5 px-4">
                      <span className={`font-semibold ${p.stock_quantity < 10 ? 'text-amber-600' : 'text-slate-800'}`}>
                        {p.stock_quantity} {p.unit}s
                      </span>
                    </td>
                    <td className="py-3.5 px-4">★ {p.rating || 5.0}</td>
                    <td className="py-3.5 px-4">
                      <span className="bg-emerald-100 text-emerald-800 text-[10px] font-bold px-2 py-0.5 rounded-full uppercase">
                        Active
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add Product Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl p-6 max-w-lg w-full space-y-4 shadow-xl border border-slate-200">
            <h3 className="text-base font-bold text-slate-900 pb-2 border-b border-slate-100">
              Add New Agricultural Product
            </h3>

            <form onSubmit={handleCreateProduct} className="space-y-3">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Product Name *</label>
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Venturi Fertigation Injector 2-inch"
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Category</label>
                  <select
                    value={categoryId}
                    onChange={(e) => setCategoryId(e.target.value)}
                    className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white"
                  >
                    {categories.map((c) => (
                      <option key={c.id} value={c.id}>{c.name}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Price (INR) *</label>
                  <input
                    type="number"
                    step="1"
                    required
                    value={price}
                    onChange={(e) => setPrice(e.target.value)}
                    className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Selling Unit</label>
                  <select
                    value={unit}
                    onChange={(e) => setUnit(e.target.value)}
                    className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white"
                  >
                    <option value="piece">piece</option>
                    <option value="meter">meter</option>
                    <option value="set">set</option>
                    <option value="kg">kg</option>
                    <option value="bag">bag</option>
                    <option value="liter">liter</option>
                    <option value="pack">pack</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Initial Stock Quantity</label>
                  <input
                    type="number"
                    value={stock}
                    onChange={(e) => setStock(e.target.value)}
                    className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Product Description</label>
                <textarea
                  rows={3}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Technical details, flow rate, compatibility, and recommended crops..."
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Photo Image URL</label>
                <input
                  type="url"
                  value={imageUrl}
                  onChange={(e) => setImageUrl(e.target.value)}
                  placeholder="https://images.unsplash.com/... (optional)"
                  className="w-full px-3 py-2 text-xs border border-slate-300 rounded-lg"
                />
              </div>

              <div className="flex justify-end space-x-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="px-4 py-2 border border-slate-300 text-slate-600 rounded-lg text-xs font-semibold"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-xs font-bold"
                >
                  Save & Publish
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProductInventory;
