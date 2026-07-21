import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import crypto from "crypto";
import fs from "fs";

dotenv.config();

const app = express();
const PORT = 3000;

// Create logs directory if missing
const LOGS_DIR = path.join(process.cwd(), "logs");
if (!fs.existsSync(LOGS_DIR)) {
  fs.mkdirSync(LOGS_DIR, { recursive: true });
}

// Security Logger
function logSecurityEvent(event: string, ip: string, details: string) {
  const timestamp = new Date().toISOString();
  const logLine = `[${timestamp}] [IP: ${ip}] [EVENT: ${event}] - ${details}\n`;
  try {
    fs.appendFileSync(path.join(LOGS_DIR, "security_audit.log"), logLine);
  } catch (err) {
    console.error("Failed to write to security audit log:", err);
  }
  console.warn(`[SECURITY AUDIT] ${event} - ${details}`);
}

// JWT Configurations
const JWT_SECRET = process.env.JWT_SECRET || crypto.randomBytes(32).toString("hex");
const revokedTokens = new Set<string>();

function signJwt(payload: any, expirySeconds: number = 3600): string {
  const header = { alg: "HS256", typ: "JWT" };
  const exp = Math.floor(Date.now() / 1000) + expirySeconds;
  const fullPayload = { ...payload, exp };
  
  const base64UrlEncode = (obj: any) => 
    Buffer.from(JSON.stringify(obj))
      .toString("base64")
      .replace(/=/g, "")
      .replace(/\+/g, "-")
      .replace(/\//g, "_");
      
  const headerEncoded = base64UrlEncode(header);
  const payloadEncoded = base64UrlEncode(fullPayload);
  
  const signature = crypto
    .createHmac("sha256", JWT_SECRET)
    .update(`${headerEncoded}.${payloadEncoded}`)
    .digest("base64url");
    
  return `${headerEncoded}.${payloadEncoded}.${signature}`;
}

function verifyJwt(token: string): any | null {
  try {
    if (revokedTokens.has(token)) return null;
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    
    const [headerEncoded, payloadEncoded, signature] = parts;
    
    const expectedSignature = crypto
      .createHmac("sha256", JWT_SECRET)
      .update(`${headerEncoded}.${payloadEncoded}`)
      .digest("base64url");
      
    if (signature !== expectedSignature) return null;
    
    const payload = JSON.parse(
      Buffer.from(payloadEncoded, "base64").toString("utf8")
    );
    
    if (payload.exp && Math.floor(Date.now() / 1000) > payload.exp) {
      return null;
    }
    
    return payload;
  } catch (err) {
    return null;
  }
}

// Custom Cookie Parser Helper
function parseCookies(cookieHeader: string | undefined): Record<string, string> {
  const list: Record<string, string> = {};
  if (!cookieHeader) return list;
  cookieHeader.split(";").forEach((cookie) => {
    const parts = cookie.split("=");
    list[parts[0].trim()] = decodeURIComponent((parts[1] || "").trim());
  });
  return list;
}

// Memory-Safe Token Bucket Rate Limiter
class TokenBucketLimiter {
  private buckets = new Map<string, { tokens: number; lastRefill: number }>();
  private maxRequests = 120;
  private refillRate = 120 / (60 * 1000);
  
  constructor() {
    setInterval(() => {
      const now = Date.now();
      for (const [ip, bucket] of this.buckets.entries()) {
        if (now - bucket.lastRefill > 10 * 60 * 1000) {
          this.buckets.delete(ip);
        }
      }
    }, 5 * 60 * 1000);
  }
  
  public limit(ip: string): boolean {
    const now = Date.now();
    let bucket = this.buckets.get(ip);
    
    if (!bucket) {
      bucket = { tokens: this.maxRequests, lastRefill: now };
      this.buckets.set(ip, bucket);
    }
    
    const elapsed = now - bucket.lastRefill;
    const refilled = elapsed * this.refillRate;
    bucket.tokens = Math.min(this.maxRequests, bucket.tokens + refilled);
    bucket.lastRefill = now;
    
    if (bucket.tokens >= 1.0) {
      bucket.tokens -= 1.0;
      return true;
    }
    return false;
  }
}

const rateLimiter = new TokenBucketLimiter();

// Gateway In-Memory Cache Registry
const gatewayCache = new Map<string, { data: any; expiry: number }>();
const inFlightRequests = new Map<string, Promise<any>>();

function getCached(key: string): any | null {
  const cached = gatewayCache.get(key);
  if (cached && Date.now() < cached.expiry) {
    return cached.data;
  }
  return null;
}

function setCached(key: string, data: any, ttlSeconds: number = 30): void {
  gatewayCache.set(key, { data, expiry: Date.now() + ttlSeconds * 1000 });
}

// 1. Strict Security Headers Middleware
app.use((req, res, next) => {
  res.setHeader("X-Frame-Options", "DENY");
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");
  res.setHeader("X-XSS-Protection", "1; mode=block");
  res.setHeader("X-Permitted-Cross-Domain-Policies", "none");
  res.setHeader("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload");
  
  res.setHeader(
    "Content-Security-Policy",
    "default-src 'self'; " +
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; " +
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; " +
    "font-src 'self' https://fonts.gstatic.com; " +
    "img-src 'self' data: blob: https://*.tile.openstreetmap.org; " +
    "connect-src 'self' ws: http://localhost:3000 http://localhost:5000 http://127.0.0.1:3000 http://127.0.0.1:5000;"
  );
  next();
});

// Strict CORS Origin Enforcement
app.use((req, res, next) => {
  const allowedOrigins = ["http://localhost:3000", "http://127.0.0.1:3000"];
  const origin = req.headers.origin;
  if (origin) {
    if (allowedOrigins.includes(origin)) {
      res.setHeader("Access-Control-Allow-Origin", origin);
    } else {
      logSecurityEvent("CORS_BLOCK", req.ip || "unknown", `Blocked origin: ${origin}`);
      return res.status(403).json({ error: "CORS policy violation." });
    }
  }
  res.setHeader("Access-Control-Allow-Methods", "GET,POST");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type,Authorization");
  res.setHeader("Access-Control-Allow-Credentials", "true");
  next();
});

app.disable("x-powered-by");
app.use(express.json({ limit: "20kb" }));

// Rate Limiter Enforcement
app.use((req, res, next) => {
  const ip = req.ip || req.socket.remoteAddress || "unknown";
  if (!rateLimiter.limit(ip)) {
    logSecurityEvent("RATE_LIMIT", ip, `Too many requests on ${req.method} ${req.url}`);
    return res.status(429).json({ error: "Too many requests. Please try again later." });
  }
  next();
});

// Cookie Session Auto-Provisioning & Verification Middleware
app.use((req: any, res, next) => {
  const cookies = parseCookies(req.headers.cookie);
  let token = cookies["session_token"];
  let user = token ? verifyJwt(token) : null;
  
  if (!user) {
    const payload = { role: "Viewer", username: "anonymous" };
    token = signJwt(payload, 86400);
    res.setHeader("Set-Cookie", `session_token=${token}; Path=/; HttpOnly; SameSite=Strict`);
    user = payload;
  }
  
  req.user = user;
  req.token = token;
  next();
});

// Initialize Gemini SDK with telemetry header
const ai = process.env.GEMINI_API_KEY
  ? new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    })
  : null;

const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://localhost:5000";

// Secure Fetch Proxy with Timeout Isolation (15s)
async function callPythonApi(path: string, method: string = "GET", body?: any) {
  const url = `${PYTHON_API_URL}${path}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15000);
  
  try {
    const options: RequestInit = {
      method,
      headers: {
        "Content-Type": "application/json",
      },
      signal: controller.signal
    };
    if (body) {
      options.body = JSON.stringify(body);
    }
    const res = await fetch(url, options);
    clearTimeout(timeoutId);
    
    if (!res.ok) {
      throw new Error(`Python Core returned HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      throw new Error("Python core connection timed out after 15 seconds.");
    }
    throw err;
  }
}

// Single-Flight Promise Cache Wrapper
async function fetchCachedData(key: string, path: string, ttlSeconds: number) {
  const cached = getCached(key);
  if (cached) return cached;
  
  let promise = inFlightRequests.get(key);
  if (!promise) {
    promise = callPythonApi(path, "GET")
      .then((data) => {
        setCached(key, data, ttlSeconds);
        inFlightRequests.delete(key);
        return data;
      })
      .catch((err) => {
        inFlightRequests.delete(key);
        throw err;
      });
    inFlightRequests.set(key, promise);
  }
  return promise;
}

// ----------------------------------------------------
// Authentication API Endpoints
// ----------------------------------------------------

let generatedAdminPass: string | null = null;
const ADMIN_USER = process.env.ADMIN_USER || "admin";
const ADMIN_PASS = process.env.ADMIN_PASS;

if (!ADMIN_PASS) {
  generatedAdminPass = crypto.randomBytes(16).toString("hex");
  console.warn(`[SECURITY WARNING] ADMIN_PASS env variable is not configured. A random administrative token has been generated for login: \n\n    >>> ADMIN PASSWORD: ${generatedAdminPass} <<<\n`);
}

app.post("/api/v1/auth/login", (req, res) => {
  const { username, password } = req.body;
  const currentPass = ADMIN_PASS || generatedAdminPass;
  
  if (username === ADMIN_USER && password && password === currentPass) {
    const token = signJwt({ role: "Admin", username }, 3600);
    const refreshToken = signJwt({ role: "Admin", username, type: "refresh" }, 7 * 86400);
    res.setHeader("Set-Cookie", [
      `session_token=${token}; Path=/; HttpOnly; SameSite=Strict`,
      `refresh_token=${refreshToken}; Path=/; HttpOnly; SameSite=Strict`
    ]);
    logSecurityEvent("AUTH_SUCCESS", req.ip || "unknown", `Admin login successful: ${username}`);
    return res.json({ status: "success", role: "Admin" });
  }
  
  logSecurityEvent("AUTH_FAIL", req.ip || "unknown", `Failed login attempt for: ${username}`);
  return res.status(401).json({ error: "Invalid username or password credentials." });
});

app.post("/api/v1/auth/logout", (req: any, res) => {
  if (req.token) {
    revokedTokens.add(req.token);
  }
  res.setHeader("Set-Cookie", [
    `session_token=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0`,
    `refresh_token=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0`
  ]);
  return res.json({ status: "logged_out" });
});

app.post("/api/v1/auth/refresh", (req, res) => {
  const cookies = parseCookies(req.headers.cookie);
  const refreshToken = cookies["refresh_token"];
  
  if (!refreshToken) {
    return res.status(401).json({ error: "Refresh token missing." });
  }
  
  const payload = verifyJwt(refreshToken);
  if (!payload || payload.type !== "refresh") {
    return res.status(401).json({ error: "Invalid or expired refresh token." });
  }
  
  const newAccessToken = signJwt({ role: payload.role, username: payload.username }, 3600);
  res.setHeader("Set-Cookie", `session_token=${newAccessToken}; Path=/; HttpOnly; SameSite=Strict`);
  return res.json({ status: "success" });
});

// Helper validation function for whitelisted station and model inputs
function validateInputs(stationId: string, modelName: string) {
  const allowedStations = [
    "sec62", "sec1", "kp3", "sec125",
    "sector-62", "sector-1", "knowledge park-iii", "sector-125",
    "sector-62, noida", "sector-1, noida", "knowledge park-iii, greater noida", "sector-125, noida"
  ];
  const cleanStation = stationId.trim().toLowerCase();
  if (!allowedStations.includes(cleanStation)) {
    throw new Error("Input station value violates security parameters bounds.");
  }
  
  const allowedModels = [
    "histgradientboosting", "histgradientboostingregressor",
    "decisiontree", "decisiontreeregressor", "randomforest", "randomforestregressor",
    "gradientboosting", "gradientboostingregressor", "xgboost",
    "linearregression", "ridgeregression", "ridge",
    "lstm", "gru", "cnn-lstm"
  ];
  const cleanModel = modelName.trim().toLowerCase();
  if (!allowedModels.includes(cleanModel)) {
    throw new Error("Input model name violates security parameters bounds.");
  }
}

// Health Check Endpoint
app.get("/api/health", async (req, res) => {
  try {
    const data = await callPythonApi("/api/health");
    res.json({
      status: "ok",
      connected: true,
      aiEnabled: !!ai,
      mlModelsLoaded: data.preloaded_estimators || [],
      lastSync: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
      environment: process.env.NODE_ENV || "development",
    });
  } catch (err: any) {
    console.error("[GATEWAY HEALTH CHECK ERROR] Python offline:", err.message);
    res.json({
      status: "error",
      connected: false,
      backend: "Express Node Gateway (Python offline)",
      aiEnabled: !!ai,
      mlModelsLoaded: [],
      lastSync: "Offline",
      environment: process.env.NODE_ENV || "development",
      error: "Could not establish handshake with Python Core.",
    });
  }
});

// Predict API
app.post("/api/predict", async (req, res) => {
  const { pollutants, meteorology, modelName, forecastHorizon, stationId } = req.body;
  
  if (!pollutants || !meteorology || !modelName || !stationId) {
    return res.status(400).json({ error: "Missing required features or inputs." });
  }
  
  try {
    validateInputs(stationId, modelName);
  } catch (err: any) {
    logSecurityEvent("VALIDATION_FAIL", req.ip || "unknown", err.message);
    return res.status(400).json({ error: err.message });
  }
  
  try {
    const data = await callPythonApi("/api/predict", "POST", req.body);
    res.json(data);
  } catch (err: any) {
    console.error("[PREDICT PATH EXCEPTION]", err.message);
    res.status(500).json({ error: "Forecasting core failed to return inference results." });
  }
});

// Forecast API
app.post("/api/forecast", async (req, res) => {
  const { pollutants, meteorology, modelName, forecastHorizon, stationId } = req.body;
  if (!pollutants || !meteorology || !modelName || !stationId) {
    return res.status(400).json({ error: "Missing required forecast inputs." });
  }
  try {
    validateInputs(stationId, modelName);
  } catch (err: any) {
    logSecurityEvent("VALIDATION_FAIL", req.ip || "unknown", err.message);
    return res.status(400).json({ error: err.message });
  }
  try {
    const data = await callPythonApi("/api/forecast", "POST", req.body);
    res.json(data);
  } catch (err: any) {
    console.error("[FORECAST PATH EXCEPTION]", err.message);
    res.status(500).json({ error: "Forecasting core failed to return forecast timeline." });
  }
});

// Models Metadata List
app.get("/api/models", async (req, res) => {
  try {
    const data = await fetchCachedData("models", "/api/models", 300);
    res.json(data);
  } catch (err: any) {
    console.error("[MODELS PATH EXCEPTION]", err.message);
    res.status(500).json({ error: "Failed to fetch model configurations." });
  }
});

// Stations Coordinate Directory
app.get("/api/stations", async (req, res) => {
  try {
    const data = await fetchCachedData("stations", "/api/stations", 300);
    res.json(data);
  } catch (err: any) {
    console.error("[STATIONS DIR PATH EXCEPTION]", err.message);
    res.status(500).json({ error: "Failed to fetch station directory." });
  }
});

// Prediction History logs
app.get("/api/prediction-history", async (req, res) => {
  try {
    const data = await callPythonApi("/api/prediction-history", "GET");
    res.json(data);
  } catch (err: any) {
    console.error("[PRED HIST EXCEPTION]", err.message);
    res.status(500).json({ error: "Failed to fetch prediction logs." });
  }
});

// Forecast History logs
app.get("/api/forecast-history", async (req, res) => {
  try {
    const data = await callPythonApi("/api/forecast-history", "GET");
    res.json(data);
  } catch (err: any) {
    console.error("[FCST HIST EXCEPTION]", err.message);
    res.status(500).json({ error: "Failed to fetch forecast logs." });
  }
});

// Health Advisory Endpoint
app.get("/api/health-advisory", async (req, res) => {
  const aqiStr = req.query.aqi ? String(req.query.aqi) : "0.0";
  try {
    const data = await fetchCachedData(`advisory_${aqiStr}`, `/api/health-advisory?aqi=${aqiStr}`, 300);
    res.json(data);
  } catch (err: any) {
    console.error("[HEALTH ADVISORY EXCEPTION]", err.message);
    res.status(500).json({ error: "Failed to fetch health advisory details." });
  }
});

// SHAP Explanations
app.post("/api/shap", async (req, res) => {
  const { pollutants, meteorology, stationId } = req.body;
  if (!pollutants || !meteorology || !stationId) {
    return res.status(400).json({ error: "Missing inputs for SHAP calculation." });
  }
  
  try {
    const allowedStations = [
      "sec62", "sec1", "kp3", "sec125",
      "sector-62", "sector-1", "knowledge park-iii", "sector-125",
      "sector-62, noida", "sector-1, noida", "knowledge park-iii, greater noida", "sector-125, noida"
    ];
    const cleanStation = stationId.trim().toLowerCase();
    if (!allowedStations.includes(cleanStation)) {
      throw new Error("Invalid station input.");
    }
  } catch (err: any) {
    return res.status(400).json({ error: err.message });
  }
  
  try {
    const data = await callPythonApi("/api/shap", "POST", req.body);
    res.json(data);
  } catch (err: any) {
    console.error("[SHAP PATH EXCEPTION]", err.message);
    res.status(500).json({ error: "Interpretability engine failed to construct attribution scores." });
  }
});

// SHAP prediction combo
app.post("/api/shap/predict", async (req, res) => {
  try {
    const data = await callPythonApi("/api/shap/predict", "POST", req.body);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "XAI Predict pipeline failed." });
  }
});

// Global XAI Feature Importance
app.get("/api/shap/global", async (req, res) => {
  try {
    const data = await fetchCachedData("shap_global", "/api/shap/global", 300);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to load global importances." });
  }
});

// Local XAI Attributions
app.get("/api/shap/local", async (req, res) => {
  try {
    const data = await callPythonApi("/api/shap/local", "GET");
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to load local attributions." });
  }
});

// Summary plot metrics
app.get("/api/shap/summary", async (req, res) => {
  try {
    const data = await callPythonApi("/api/shap/summary", "GET");
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to load summary plot." });
  }
});

// Beeswarm plot metrics
app.get("/api/shap/beeswarm", async (req, res) => {
  try {
    const data = await callPythonApi("/api/shap/beeswarm", "GET");
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to load beeswarm plot." });
  }
});

// Waterfall plot metrics
app.get("/api/shap/waterfall", async (req, res) => {
  try {
    const data = await callPythonApi("/api/shap/waterfall", "GET");
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to load waterfall plot." });
  }
});

// Force plot metrics
app.get("/api/shap/force", async (req, res) => {
  try {
    const data = await callPythonApi("/api/shap/force", "GET");
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to load force plot." });
  }
});

// Dependence plot metrics
app.get("/api/shap/dependence", async (req, res) => {
  try {
    const data = await callPythonApi("/api/shap/dependence", "GET");
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to load dependence plot." });
  }
});

// Decision plot metrics
app.get("/api/shap/decision", async (req, res) => {
  try {
    const data = await callPythonApi("/api/shap/decision", "GET");
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to load decision plot." });
  }
});

// Stations Telemetry Data
app.get("/api/stations-data", async (req, res) => {
  try {
    const data = await fetchCachedData("stations_data", "/api/stations-data", 30);
    res.json(data);
  } catch (err: any) {
    console.error("[STATIONS PATH EXCEPTION]", err.message);
    res.status(500).json({ error: "Failed to fetch station telemetry records." });
  }
});

// Historical Calendar Trends
app.get("/api/historical-trends", async (req, res) => {
  const yearStr = req.query.year ? String(req.query.year) : "2024";
  if (!["2024", "2025", "2026"].includes(yearStr)) {
    return res.status(400).json({ error: "Invalid year parameter. Must be 2024, 2025 or 2026." });
  }
  try {
    const data = await fetchCachedData(`historical_trends_${yearStr}`, `/api/historical-trends?year=${yearStr}`, 60);
    res.json(data);
  } catch (err: any) {
    console.error("[TRENDS PATH EXCEPTION]", err.message);
    res.status(500).json({ error: "Failed to assemble multi-year seasonal aggregates." });
  }
});

// EDA correlations
app.get("/api/eda/correlations", async (req, res) => {
  try {
    const data = await fetchCachedData("eda_correlations", "/api/eda/correlations", 300);
    res.json(data);
  } catch (err: any) {
    console.error("[EDA CORRELATION EXCEPTION]", err.message);
    res.status(500).json({ error: "Failed to load Pearson correlation matrix." });
  }
});

// Dashboard summary
app.get("/api/dashboard", async (req, res) => {
  try {
    const data = await fetchCachedData("dashboard", "/api/dashboard", 10);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to load dashboard metrics." });
  }
});

// Overview stats
app.get("/api/overview", async (req, res) => {
  try {
    const data = await fetchCachedData("overview", "/api/overview", 10);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to load overview telemetry." });
  }
});

// Current concentrations
app.get("/api/current", async (req, res) => {
  try {
    const data = await fetchCachedData("current", "/api/current", 10);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to load active concentrations." });
  }
});

// Analytics averages
app.get("/api/analytics", async (req, res) => {
  try {
    const data = await fetchCachedData("analytics", "/api/analytics", 30);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to compile analytical trends." });
  }
});

// Exploratory Data Analysis metrics
app.get("/api/eda", async (req, res) => {
  try {
    const data = await fetchCachedData("eda", "/api/eda", 300);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to compute exploratory metrics." });
  }
});

// Map markers
app.get("/api/map", async (req, res) => {
  try {
    const data = await fetchCachedData("map", "/api/map", 10);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to locate spatial mapping nodes." });
  }
});

// Persistent prediction logs
app.get("/api/history", async (req, res) => {
  try {
    const data = await callPythonApi("/api/history", "GET");
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to retrieve history logs." });
  }
});

// Gateway System status
app.get("/api/system-status", async (req, res) => {
  try {
    const data = await fetchCachedData("system_status", "/api/system-status", 10);
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to fetch hardware runtime status." });
  }
});

// Explain Predictions via Gemini
app.post("/api/gemini/explain-prediction", async (req, res) => {
  const { pollutants, meteorology, station, predictedAqi, forecastHorizon, modelName } = req.body;
  if (!pollutants || !meteorology || !station || predictedAqi === undefined) {
    return res.status(400).json({ error: "Missing telemetry indicators." });
  }
  
  // Prompt Injection Mitigation: Strict input parameter whitelisting
  try {
    validateInputs(station, modelName);
  } catch (err: any) {
    logSecurityEvent("PROMPT_INJECTION_SHIELD", req.ip || "unknown", `Blocked malicious prompt input: ${station} / ${modelName}`);
    return res.status(400).json({ error: "Invalid prompt payload parameters." });
  }
  
  if (!ai) {
    const primaryFactor = pollutants.pm25 > 100 ? "PM2.5" : pollutants.no2 > 80 ? "NO2" : "PM10";
    const recommendation = predictedAqi > 150 
      ? "Active air purification recommended. Sensitive groups should avoid all outdoor activities."
      : "Acceptable air quality. Standard outdoor activities can continue.";
    
    return res.json({
      summary: `The forecasting engine predicts an AQI of **${predictedAqi}** over the next **${forecastHorizon} hours** at **${station}** using the **${modelName}** production model.`,
      analysis: `The primary driver of this trend is **${primaryFactor}** concentration (${pollutants.pm25} µg/m³), aggravated by meteorological conditions including low wind speed (${meteorology.windSpeed} km/h) and moderate humidity (${meteorology.humidity}%), which inhibit pollutant dispersion.`,
      recommendation: recommendation,
      source: "Local rule-based analytical fallback (configure GEMINI_API_KEY in Secrets for advanced AI insights)."
    });
  }

  try {
    const prompt = `You are a Senior Atmospheric Scientist and Explainable AI Specialist. Analyze this air quality forecasting scenario:
Station: ${station}
Forecast Model: ${modelName}
Forecast Horizon: ${forecastHorizon} hours
Predicted AQI: ${predictedAqi}

Current/Input Pollutants:
- PM2.5: ${pollutants.pm25} µg/m³
- PM10: ${pollutants.pm10} µg/m³
- NO2: ${pollutants.no2} ppb
- SO2: ${pollutants.so2} ppb
- CO: ${pollutants.co} ppm
- O3: ${pollutants.o3} ppb
- NH3: ${pollutants.nh3} ppb

Meteorological Factors:
- Temperature: ${meteorology.temperature}°C
- Humidity: ${meteorology.humidity}%
- Wind Speed: ${meteorology.windSpeed} km/h
- Rainfall: ${meteorology.rainfall} mm

Based on this, provide a concise, research-grade, professional analysis in 3 sections (Format as valid JSON with "summary", "analysis", "recommendation" keys). 
Use scientific tone, explain atmospheric interactions (e.g., thermal inversion, chemical reactions, dispersion, photochemistry of ozone if solar radiation/temperature is high). Keep markdown formatting elegant. Do not use conversational filler.`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
      }
    });

    const text = response.text || "{}";
    try {
      const parsed = JSON.parse(text);
      res.json({
        summary: parsed.summary || "",
        analysis: parsed.analysis || "",
        recommendation: parsed.recommendation || "",
        source: "Gemini 3.5 Flash Model Insights"
      });
    } catch (parseErr) {
      res.json({
        summary: `Forecast predicted AQI is ${predictedAqi}.`,
        analysis: text,
        recommendation: "Take standard precautionary measures for this AQI level.",
        source: "Gemini 3.5 Flash Model Raw Response"
      });
    }
  } catch (err: any) {
    console.error("[GEMINI EXPLAIN EXCEPTION]", err.message);
    res.status(500).json({ error: "AI explanation model service is currently unavailable." });
  }
});

// Explain SHAP via Gemini
app.post("/api/gemini/explain-shap", async (req, res) => {
  const { topFeatures, baseValue, predictedValue } = req.body;
  if (!topFeatures || baseValue === undefined || predictedValue === undefined) {
    return res.status(400).json({ error: "Missing attributions coordinates." });
  }

  // Prompt Injection Mitigation: Validate array and structure content values
  if (!Array.isArray(topFeatures)) {
    return res.status(400).json({ error: "Invalid features format." });
  }
  const safeRegex = /^[a-zA-Z0-9_\-\.\s\(\)\+]+$/;
  for (const f of topFeatures) {
    if (typeof f.name !== "string" || typeof f.value !== "number" || typeof f.featureValue !== "string") {
      return res.status(400).json({ error: "Invalid feature format types." });
    }
    if (!safeRegex.test(f.name) || !safeRegex.test(f.featureValue)) {
      logSecurityEvent("PROMPT_INJECTION_SHIELD", req.ip || "unknown", `Blocked malicious SHAP prompt text: ${f.name}`);
      return res.status(400).json({ error: "Malicious characters detected in prompt parameters." });
    }
  }

  if (!ai) {
    return res.json({
      explanation: `Local SHAP attribution explains how features shifted the prediction from the baseline of **${baseValue}** to the final forecast of **${predictedValue}**. The largest positive driver is **${topFeatures[0]?.name || "PM2.5"}** (+${topFeatures[0]?.value?.toFixed(1) || 24.5} AQI impact), while the largest counteracting feature is **${topFeatures[topFeatures.length - 1]?.name || "Wind Speed"}** (${topFeatures[topFeatures.length - 1]?.value?.toFixed(1) || -5.2} AQI impact).`
    });
  }

  try {
    const prompt = `You are an AI interpretability researcher. Explain these SHAP (SHapley Additive exPlanations) values for an air quality forecasting model:
Model Base Value (f(x) average baseline): ${baseValue}
Model Output Prediction: ${predictedValue}

Top contributing features and their SHAP values (positive values pushed the prediction up, negative values pulled it down):
${topFeatures.map((f: any) => `- ${f.name}: SHAP value = ${f.value > 0 ? "+" : ""}${f.value.toFixed(2)} (Actual Feature Value: ${f.featureValue})`).join("\n")}

Provide an elegant explanation (2-3 sentences) summarizing the mathematical alignment and physical/chemical reason why these features had these effects on the final forecast. Output your explanation inside a JSON object with key "explanation".`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
      }
    });

    const text = response.text || "{}";
    try {
      const parsed = JSON.parse(text);
      res.json({
        explanation: parsed.explanation || text
      });
    } catch (parseErr) {
      res.json({
        explanation: text
      });
    }
  } catch (err: any) {
    console.error("[GEMINI SHAP EXCEPTION]", err.message);
    res.status(500).json({ error: "AI explainability model service is currently offline." });
  }
});

// Proxy Secure Download Endpoints
app.get("/api/download/:file", async (req, res) => {
  const file = req.params.file;
  
  const allowedFiles = ["paper", "dataset", "model", "shap", "figures", "results"];
  if (!allowedFiles.includes(file)) {
    logSecurityEvent("PATH_TRAVERSAL_PREVENTED", req.ip || "unknown", `Unauthorized download target file: ${file}`);
    return res.status(400).json({ error: "Invalid download selection." });
  }
  
  try {
    const pythonUrl = `${PYTHON_API_URL}/api/download/${file}`;
    const response = await fetch(pythonUrl);
    if (!response.ok) {
      return res.status(response.status).send("Requested asset not found on core storage.");
    }
    
    response.headers.forEach((val, key) => {
      res.setHeader(key, val);
    });
    
    const arrayBuffer = await response.arrayBuffer();
    res.send(Buffer.from(arrayBuffer));
  } catch (err: any) {
    console.error("[DOWNLOAD PATH EXCEPTION]", err.message);
    res.status(500).json({ error: "Failed to download requested publication/dataset asset." });
  }
});

// Start server hosting
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath, {
      setHeaders: (res) => {
        res.setHeader("X-Frame-Options", "DENY");
        res.setHeader("X-Content-Type-Options", "nosniff");
      }
    }));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }
  
  app.listen(PORT, () => {
    console.log(`[AQI Backend] Secure Enterprise Server listening at http://localhost:${PORT}`);
  });
}

startServer();
