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

## 3. Production Deployment with PostgreSQL

### Environment Variables
Set the following environment variables in production:
```ini
FLASK_ENV=production
SECRET_KEY=generate-a-strong-random-hex-key
JWT_SECRET_KEY=generate-a-strong-jwt-secret-key
DATABASE_URL=postgresql://agri_user:agri_password@postgres-host:5432/smart_irrigation_prod
PORT=5000
```

### Production WSGI Server (Gunicorn)
```bash
cd backend
pip install gunicorn psycopg2-binary
gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app()"
```

### Production Frontend Build
```bash
cd frontend
npm run build
# Deploy the generated dist/ folder to Nginx, AWS S3/CloudFront, or Vercel
```
