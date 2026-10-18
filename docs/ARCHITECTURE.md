# Smart Irrigation Architecture & Technical Design

This document describes the architectural layers, database models, ML pipeline, and security design of the **Smart Irrigation & Agriculture Lifecycle Management Platform**.

---

## 1. High-Level System Architecture

```mermaid
graph TD
    Client["React 18 + Vite + Tailwind Frontend"]
    
    subgraph "Backend API (Flask REST /api/v1)"
        Auth["Auth & RBAC (Flask-JWT-Extended)"]
        WaterEng["Water Balance Engine (FAO-56)"]
        MLEng["ML Pipeline (Scikit-Learn LinearRegression)"]
        Weather["Weather Service (Open-Meteo Cache)"]
        Market["Marketplace & Atomic Checkout"]
        Consult["Consultation Threading"]
    end
    
    subgraph "Data Storage"
        DB[("Normalized Database (SQLite / PostgreSQL)")]
        Models[("Serialized Joblib Pipeline")]
        CSV[("Dataset (1,500 Records CSV)")]
    end
    
    Client -->|"HTTP/JSON with JWT"| Auth
    Client --> Weather
    Client --> WaterEng
    Client --> MLEng
    Client --> Market
    Client --> Consult
    
    WaterEng --> DB
    MLEng --> Models
    Weather --> DB
    Market --> DB
    Consult --> DB
```

---

## 2. Machine Learning Pipeline Architecture

The continuous irrigation prediction problem uses a scikit-learn pipeline:

```mermaid
graph LR
    Raw["Input Features (Categorical & Numerical)"]
    
    subgraph "ColumnTransformer"
        Cat["soil_type, crop_type, crop_growth_stage"] --> OHE["OneHotEncoder(handle_unknown='ignore')"]
        Num["temp, humidity, moisture, rain, wind, solar, ET0"] --> SS["StandardScaler()"]
    end
    
    OHE --> Reg["LinearRegression() Regressor"]
    SS --> Reg
    Reg --> Clip["Non-negative Truncation max(0, y)"]
    Clip --> Output["Net Irrigation Requirement (mm)"]
```

### Model Performance Metrics (Held-Out Test Set):
- **Algorithm**: Ordinary Least Squares Linear Regression with Preprocessing Pipeline
- **Dataset Size**: 1,500 agronomically modeled records
- **Train / Test Split**: 80% (1,200 rows) / 20% (300 rows) with random seed = 42
- **Mean Absolute Error (MAE)**: 1.78 mm
- **Root Mean Squared Error (RMSE)**: 2.76 mm
- **$R^2$ Score**: ~0.70

---

## 3. Database Entity Relationship (ER) Summary

The database is designed with normalized tables adhering to 3NF:
- **`users`**: Core user accounts (email, password_hash, role, phone).
- **`farmer_profiles`**, **`specialist_profiles`**, **`retailer_profiles`**: Role-specific polymorphic profile tables.
- **`farms`** & **`farm_plots`**: Multi-plot farm hierarchies with geolocation coordinates and soil classifications.
- **`crop_references`**: Agronomic parameters ($K_c$ stages, temperature windows, rooting depth, irrigation method).
- **`irrigation_predictions`**: Complete audit snapshot of input weather/soil data, ML predictions, and gross irrigation calculations.
- **`consultations`** & **`consultation_messages`**: Threaded Q&A with specialist diagnostic badges and feedback ratings.
- **`products`**, **`carts`**, **`cart_items`**, **`orders`**, **`order_items`**: Complete e-commerce loop with transactional stock validation.
- **`quotation_requests`** & **`quotations`**: Farmer bulk RFQ bidding system.
- **`audit_logs`**: System audit trail tracking sensitive actions.

---

## 4. Security & Reliability

1. **Password Security**: Werkzeug `generate_password_hash` using salted SHA-256.
2. **Stateless JWT**: Standard Bearer authorization header with 7-day token expiration.
3. **Role-Based Access Control**: Backend `@role_required` decorator checks role on every sensitive endpoint.
4. **Atomic Transactions**: All marketplace order creations execute within database transactions (`db.session.rollback()` upon insufficient stock).
5. **CORS & Input Validation**: Strict validation prevents physically unrealistic measurements (e.g. negative rainfall or moisture > 100%).
