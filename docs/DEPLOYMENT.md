# Smart Irrigation Platform – Deployment & Operations Guide

## 1. Local Development Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm 9+
- Git

### Quick Setup

#### Step 1: Clone or navigate to the repository
```bash
cd "C:\Users\anupam kumar\.gemini\antigravity\scratch\smart-irrigation"
```

#### Step 2: Set up the Python Backend & Machine Learning Model
```bash
# Optional: create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install backend dependencies
cd backend
pip install -r requirements.txt

# (Re)generate dataset & train the ML pipeline
cd ../ml
python generate_dataset.py
python train.py
python evaluate.py

# Run backend test suite
cd ../backend
python -m pytest tests/ -v

# Start the Flask Backend Server (Port 5000)
python run.py
```

#### Step 3: Set up the React Frontend
Open a second terminal window:
```bash
cd "C:\Users\anupam kumar\.gemini\antigravity\scratch\smart-irrigation\frontend"
npm install
npm run dev
```
Access the application at `http://localhost:5173`.

---

## 2. Default Test Accounts (Pre-Seeded)

For immediate local evaluation across all 5 roles without needing manual registration:

| Persona | Role | Email | Password |
|---|---|---|---|
| **Farmer** | `farmer` | `farmer@smartirrigation.org` | `Farmer@123` |
| **Retailer** | `retailer` | `retailer@smartirrigation.org` | `Retailer@123` |
| **Customer** | `customer` | `customer@smartirrigation.org` | `Customer@123` |
| **Specialist** | `specialist` | `specialist@smartirrigation.org` | `Specialist@123` |
| **Administrator** | `admin` | `admin@smartirrigation.org` | `Admin@123` |

*Note: The platform features a 1-click test login bar at the top of the screen to switch personas effortlessly.*

---

## 3. Cloud Deployment Options

### Option A: Render 1-Click Deployment (Recommended Free Cloud Hosting)

The repository includes a ready-to-run [render.yaml](../render.yaml) blueprint and multi-stage [Dockerfile](../Dockerfile).

1. Sign up / Log in to [Render](https://render.com).
2. Go to the **Dashboard** and click **New +** > **Blueprint**.
3. Connect your GitHub account and select repository: `https://github.com/anupamkr2307/Smart-irrigation`.
4. Render will detect `render.yaml` and configure:
   - **Environment**: Docker (multi-stage build compiling React and running Python 3.12 Gunicorn).
   - **Plan**: Free.
   - **Auto-generated Secrets**: `SECRET_KEY` and `JWT_SECRET_KEY`.
5. Click **Apply**.
6. Render builds both the React client and Python backend, issuing a free live HTTPS URL (e.g. `https://smart-irrigation.onrender.com`).

*(Optional)* To attach a persistent cloud PostgreSQL database on Render:
- Go to **New +** > **PostgreSQL**.
- Copy the **Internal Database URL** and set it as `DATABASE_URL` under your web service **Environment** tab.

---

### Option B: Split Deployment (Frontend on Vercel + Backend on Render)

If you prefer hosting the React frontend on Vercel's global CDN:

#### 1. Backend on Render
- Deploy the backend as a Docker or Python Web Service on Render.
- Note your live API URL: `https://your-api.onrender.com`.

#### 2. Frontend on Vercel
1. Log in to [Vercel](https://vercel.com) and click **Add New...** > **Project**.
2. Select `https://github.com/anupamkr2307/Smart-irrigation`.
3. In the project settings:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Add the Environment Variable:
   - `VITE_API_URL` = `https://your-api.onrender.com`
5. Click **Deploy**. Vercel will deploy using `frontend/vercel.json` for seamless client-side SPA routing.

---

### Option C: Containerized Self-Hosted Deployment (Docker & Docker Compose)

Deploy the full stack locally or on any Linux VPS (Ubuntu, Debian, AWS EC2, DigitalOcean Droplet, Linode) with Docker:

#### 1. Start with Docker Compose (includes PostgreSQL)
```bash
git clone https://github.com/anupamkr2307/Smart-irrigation.git
cd Smart-irrigation

docker compose up --build -d
```
The stack will build the multi-stage image, spin up PostgreSQL 16, run health checks, initialize database tables, seed test accounts, and expose the app on port `5000`.

#### 2. Verify containers
```bash
docker compose ps
docker compose logs -f web
```
Access the application at `http://localhost:5000` (or `http://your-server-ip:5000`).

#### 3. Stop containers
```bash
docker compose down
```

---

## 4. Continuous Integration & Deployment (CI/CD)

The repository includes a GitHub Actions workflow in [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml):
- **On Push to `main`**: Automatically sets up Python 3.12, installs dependencies, runs the 18 Pytest unit tests, and verifies the React Vite production build.

