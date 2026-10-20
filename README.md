# Smart Irrigation – Complete Agriculture Solution
### AI-Powered Agriculture Lifecycle Management Platform

[![Live Demo (Vercel)](https://img.shields.io/badge/Live%20Demo-Vercel%20(Online)-success?style=flat&logo=vercel)](https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/)
[![Live Demo (GitHub Pages)](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages%20(Lifetime)-success?style=flat&logo=github)](https://anupamkr2307.github.io/Smart-irrigation/)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fanupamkr2307%2FSmart-irrigation)
[![Backend Tests](https://img.shields.io/badge/pytest-21%20passed-brightgreen.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/ML-LinearRegression%20Pipeline-blue.svg)]()
[![FAO-56 Standard](https://img.shields.io/badge/Standard-FAO--56%20Penman--Monteith-emerald.svg)]()
[![License](https://img.shields.io/badge/License-MIT-slate.svg)]()

A complete, modern, interactive, full-stack web application designed to help farmers make data-driven decisions about irrigation, crop selection, rainfall requirements, soil management, and agricultural commerce.

The platform unifies **5 integrated personas**: Farmers, Retailers, Customers, Agricultural Specialists, and Administrators with dedicated role registration portals and permission-enforced authorization.

---

## 🌐 Lifetime Live Links & Instant Access

> 🚀 **Explore the live platform online**:
>
> | Platform | Deployment Type | Permanent Live Link | Status |
> |---|---|---|---|
> | **Vercel** | **Primary Cloud Deployment** | [https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/](https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/) | `Live Online` |
> | **GitHub Pages** | **Lifetime 100% Free** | [https://anupamkr2307.github.io/Smart-irrigation/](https://anupamkr2307.github.io/Smart-irrigation/) | `Active CI/CD` |
> | **Render** | **Full-Stack Cloud Web Service** | [render.yaml Blueprint](render.yaml) | `Ready` |
> | **Active Live Tunnel** | **Interactive Live Demo** | [https://wxchp-152-58-25-20.free.pinggy.net](https://wxchp-152-58-25-20.free.pinggy.net) | `200 OK Online` |
> | **Tunnel Mirror** | **Interactive Mirror** | [https://olzze-152-58-25-20.run.pinggy-free.link](https://olzze-152-58-25-20.run.pinggy-free.link) | `200 OK Online` |
>
> - **Live Web Application**: [https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/](https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/)
> - **Multi-Role Registration Portal**: [https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/register](https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/register)
> - **Role-Enforced Login Portal**: [https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/login](https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/login)
> - **Interactive Swagger / OpenAPI Docs**: [https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/api/v1/docs](https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/api/v1/docs)
> - **Role Capabilities Metadata**: [https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/api/v1/auth/roles](https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/api/v1/auth/roles)
> - **Backend Health Check**: [https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/api/health](https://smart-irrigation-1z31-4zfh5ytzl-anupamkr2307.vercel.app/api/health)
>
> *Use the 1-click **Quick Demo Bar** docked at the top of the browser to instantly test all 5 user roles without typing credentials!*

---

## ⚡ 1-Click Deploy to Vercel

You can deploy this full-stack application directly to **Vercel** with one click:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fanupamkr2307%2FSmart-irrigation)

### Deployment Methods:

#### Option A: 1-Click GitHub Import (Recommended)
1. Visit [vercel.com/new](https://vercel.com/new).
2. Choose **Import Git Repository** and select `anupamkr2307/Smart-irrigation`.
3. Keep default settings:
   - **Framework Preset**: Vite
   - **Root Directory**: `./` (leave default, `vercel.json` configures everything)
4. Click **Deploy**. Vercel will build the frontend into `frontend/dist` and deploy the Python backend as Serverless Functions (`api/index.py`)!

#### Option B: Deploy via Vercel CLI
```bash
# 1. Login to Vercel
npx vercel login

# 2. Deploy directly to production
npx vercel --prod
```



## Key Highlights

- **Linear Regression ML Pipeline:** Predicts continuous net irrigation requirement (mm) using scikit-learn `Pipeline` (`OneHotEncoder` + `StandardScaler` + `LinearRegression`), trained on 1,500 structured records ($R^2 \approx 0.70$, MAE $1.78\text{ mm}$).
- **FAO-56 Transparent Water Balance:** Computes crop evapotranspiration ($ET_c = K_c \times ET_0$), USDA-SCS effective rainfall, usable soil moisture, and gross irrigation volume accounting for system efficiency (Drip 90%, Sprinkler 75%, Furrow 60%).
- **Open-Meteo & Leaflet Integration:** Real-time geocoding, temperature, humidity, and 7-day rainfall forecasts with interactive OpenStreetMap parcel selection and manual fallbacks.
- **Multi-Factor Crop Suitability Engine:** Evaluates soil texture, pH, seasonal temperature windows, and water availability to recommend the top 5 viable crops.
- **Farmer–Specialist Tele-Advisory Desk:** Threaded diagnostics with certified specialist badges, image attachments, and 5-star resolution ratings.
- **Agricultural Marketplace & Quotations:** Retailer store management, atomic stock validation checkout with test mode payment, and farmer bulk quotation bidding.
- **Admin Governance & Dataset Inspector:** User management, specialist verification toggle, ML metrics analyzer, and full 1,500-record dataset table.

---

## Technology Stack

- **Frontend:** React 18, Vite, Tailwind CSS, React Router v6, Axios, Recharts, Lucide React, Leaflet & React-Leaflet, React Hot Toast.
- **Backend:** Python 3.13, Flask 3.0, Flask-SQLAlchemy, Flask-Migrate, Flask-JWT-Extended, Flask-CORS, Werkzeug.
- **Machine Learning:** scikit-learn, pandas, NumPy, Joblib.
- **Database:** SQLite for local development, PostgreSQL compatible schema.

---

## Pre-Configured Test Accounts (1-Click Switcher)

| Role | Email | Password | Details |
|---|---|---|---|
| **Farmer** | `farmer@smartirrigation.org` | `Farmer@123` | Nashik vineyard & plots, runs irrigation predictions |
| **Retailer** | `retailer@smartirrigation.org` | `Retailer@123` | Kisan Seva Kendra shop, inventory & orders |
| **Customer** | `customer@smartirrigation.org` | `Customer@123` | Direct equipment & supply checkout |
| **Specialist** | `specialist@smartirrigation.org` | `Specialist@123` | Dr. Arvind Deshmukh, verified agronomist |
| **Administrator** | `admin@smartirrigation.org` | `Admin@123` | System oversight, ML analytics, audit logs |

*A demo access bar is docked at the top of the app to switch between roles instantly.*

---

## Quickstart Guide

### 1. Start the Flask Backend
```bash
cd backend
python -m pip install -r requirements.txt
python run.py
```
*The backend server will start on `http://localhost:5000` and automatically create the database and seed initial test accounts.*

### 2. Start the React Frontend
```bash
cd frontend
npm install
npm run dev
```
*The frontend will run on `http://localhost:5173` with an automatic proxy to the Flask API.*

### 3. Run Automated Tests
```bash
cd backend
python -m pytest tests/ -v
```

---

## Directory Structure

```
smart-irrigation/
├── backend/
│   ├── app/
│   │   ├── config.py              # SQLite / Postgres configuration
│   │   ├── extensions.py          # SQLAlchemy, JWT, CORS, Migrate
│   │   ├── models/                # Normalized database entities
│   │   ├── routes/                # REST API blueprints (/api/v1/...)
│   │   ├── services/              # Water balance, ML inference, weather proxy
│   │   └── utils/                 # RBAC decorators and database seeder
│   ├── tests/                     # 18 automated Pytest suites
│   ├── requirements.txt
│   └── run.py
├── frontend/
│   ├── src/
│   │   ├── components/            # MapPicker, Navbar, Sidebar, StatCard, DemoBar
│   │   ├── context/               # AuthContext (5 roles), UnitContext (Acres/Ha)
│   │   ├── pages/                 # Public, Farmer, Retailer, Customer, Specialist, Admin
│   │   └── services/api.js        # Axios instance with Bearer interceptor
│   ├── package.json
│   └── vite.config.js
├── ml/
│   ├── generate_dataset.py        # Generates 1,500 structured records
│   ├── preprocess.py              # Data cleaning and bounds validation
│   ├── train.py                   # Scikit-learn Pipeline training & evaluation
│   ├── evaluate.py                # Model diagnostics
│   └── models/                    # Serialized model (.joblib) & metrics (.json)
├── dataset/
│   ├── irrigation_data.csv        # 1,500 structured records with 23 fields
│   └── dataset_metadata.json      # Agronomic assumptions & variable dictionary
└── docs/                          # Architecture, Formulas, API & Deployment guides
```

---

## Documentation Links

- [Agronomic Formulas & Physics](docs/AGRONOMIC_FORMULAS.md)
- [REST API Reference](docs/API_DOCUMENTATION.md)
- [System Architecture & ML Design](docs/ARCHITECTURE.md)
- [Deployment & PostgreSQL Guide](docs/DEPLOYMENT.md)
