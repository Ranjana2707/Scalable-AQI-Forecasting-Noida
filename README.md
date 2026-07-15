# Noida AQI Forecasting & Explainable AI (XAI) System

An enterprise-grade, real-time Air Quality Index (AQI) forecasting and Explainable AI platform. It utilizes machine learning models (HistGradientBoosting, DecisionTree, GradientBoosting) and deep learning networks (LSTM, GRU, CNN-LSTM) to perform dynamic inference and SHAP local interpretability.

---

## Repository Structure

- `/gateway`: Node.js Express API gateway (implements authentication, rate limiting, and single-flight caching proxy).
- `/backend`: Python Flask XAI Core (handles model loading, inference execution, and SHAP computations).
- `/docker-compose.yml`: Orchestrates production containers on port 3000.
- `/.env.example`: Template for environment keys and credentials.

---

## Local Development Setup

### Prerequisite 1: Start Python Core Backend
1. Navigate to `/backend`.
2. Create python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install package requirements:
   ```bash
   pip install -r requirements.txt
   pip install flask flask-cors gunicorn
   ```
4. Start the server:
   ```bash
   python server_api.py
   ```
   Flask listens at `http://localhost:5000`.

### Prerequisite 2: Start API Gateway
1. Navigate to `/gateway`.
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Create `.env` file using `.env.example` as a template.
4. Bundle and run the gateway:
   ```bash
   npm run build
   npm run dev
   ```
   The gateway listens at `http://localhost:3000`.

---

## Production Docker Deployment

Deploy both layers inside isolated network containers:

1. Configure `.env` variables at the root level.
2. Build and launch:
   ```bash
   docker-compose up --build -d
   ```
3. Access the dashboard UI on `http://localhost:3000`.
