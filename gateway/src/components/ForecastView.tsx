import React, { useState, useEffect } from "react";
import { STATIONS } from "../data";
import { Sliders, Wind, Droplets, Thermometer, CloudRain, ShieldCheck, Sparkles, AlertCircle, RefreshCw, Layers } from "lucide-react";

interface ForecastViewProps {
  selectedStation: string;
  setSelectedStation: (id: string) => void;
  predictedAqi: number;
  setPredictedAqi: (val: number) => void;
  pollutants: { pm25: number; pm10: number; no2: number; so2: number; co: number; o3: number; nh3: number };
  setPollutants: React.Dispatch<React.SetStateAction<{ pm25: number; pm10: number; no2: number; so2: number; co: number; o3: number; nh3: number }>>;
  meteorology: { temperature: number; humidity: number; windSpeed: number; rainfall: number };
  setMeteorology: React.Dispatch<React.SetStateAction<{ temperature: number; humidity: number; windSpeed: number; rainfall: number }>>;
  forecastHorizon: number;
  setForecastHorizon: (val: number) => void;
}

export default function ForecastView({
  selectedStation,
  setSelectedStation,
  predictedAqi,
  setPredictedAqi,
  pollutants,
  setPollutants,
  meteorology,
  setMeteorology,
  forecastHorizon,
  setForecastHorizon
}: ForecastViewProps) {
  const [activeModel, setActiveModel] = useState("HistGradientBoosting");
  const [sensorDrift, setSensorDrift] = useState(false);
  const [confidenceInterval, setConfidenceInterval] = useState(true);
  const [historical, setHistorical] = useState<number[]>([142, 148, 156, 150, 162, 175, 184]);
  const [forecast, setForecast] = useState<number[]>([190, 195, 204, 210, 220, 230]);
  const [backendLoading, setBackendLoading] = useState(false);
  const [backendError, setBackendError] = useState(false);

  // Time & Date State Selection
  const [selectedDate, setSelectedDate] = useState(() => {
    const d = new Date();
    return d.toISOString().split("T")[0];
  });
  const [selectedTime, setSelectedTime] = useState(() => {
    const d = new Date();
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    return `${hours}:${minutes}`;
  });
  const [predictionHistory, setPredictionHistory] = useState<any[]>([
    { id: 1, timestamp: "2026-07-10 18:00", station: "Sector-62, Noida", predictedAqi: 210, category: "Poor", model: "HistGradientBoosting" },
    { id: 2, timestamp: "2026-07-10 12:00", station: "Sector-1, Noida", predictedAqi: 184, category: "Moderate", model: "HistGradientBoosting" }
  ]);

  // AI Explanation State
  const [aiLoading, setAiLoading] = useState(false);
  const [aiReport, setAiReport] = useState<{ summary: string; analysis: string; recommendation: string; source: string } | null>(null);

  // Fetch forecast prediction from server when covariates or sliders change
  useEffect(() => {
    let active = true;
    const controller = new AbortController();

    async function fetchForecast() {
      setBackendLoading(true);
      setBackendError(false);
      try {
        const bodyPollutants = { ...pollutants };
        if (sensorDrift) {
          bodyPollutants.pm25 += 25;
        }

        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pollutants: bodyPollutants,
            meteorology,
            modelName: activeModel,
            forecastHorizon,
            stationId: selectedStation,
            date: selectedDate,
            time: selectedTime
          }),
          signal: controller.signal
        });
        const data = await res.json();
        if (active) {
          if (data.predictedAqi !== undefined) {
            setPredictedAqi(data.predictedAqi);
            setHistorical(data.historical);
            setForecast(data.forecast);

            // Append to history log
            setPredictionHistory(prev => {
              const currentStationName = STATIONS.find(s => s.id === selectedStation)?.name || "Sector-62, Noida";
              const newEntry = {
                id: Date.now(),
                timestamp: `${selectedDate} ${selectedTime}`,
                station: currentStationName,
                predictedAqi: data.predictedAqi,
                category: getCategoryTheme(data.predictedAqi).label,
                model: activeModel
              };
              const isDuplicate = prev.some(
                item => item.timestamp === newEntry.timestamp && 
                        item.station === newEntry.station && 
                        item.predictedAqi === newEntry.predictedAqi &&
                        item.model === newEntry.model
              );
              if (isDuplicate) return prev;
              return [newEntry, ...prev.slice(0, 4)];
            });
          } else {
            setBackendError(true);
          }
        }
      } catch (err: any) {
        if (err.name !== "AbortError") {
          console.error("Prediction API fetch failed:", err);
          if (active) setBackendError(true);
        }
      } finally {
        if (active) setBackendLoading(false);
      }
    }

    fetchForecast();

    return () => {
      active = false;
      controller.abort();
    };
  }, [pollutants, meteorology, activeModel, sensorDrift, forecastHorizon, selectedStation, selectedDate, selectedTime, setPredictedAqi]);

  const getHealthAdvisory = (aqi: number) => {
    if (aqi <= 50) return "Standard outdoor activities can continue safely. Atmospheric conditions pose negligible health risk.";
    if (aqi <= 100) return "Acceptable air quality. Standard outdoor activities can continue. Extremely sensitive groups should monitor symptoms.";
    if (aqi <= 200) return "Sensitive groups may experience health effects. Restrict prolonged or heavy outdoor exertion.";
    if (aqi <= 300) return "Avoid outdoor exertion. Wear particulate masks. Active indoor air purification recommended.";
    if (aqi <= 400) return "Significant health impact. Stay indoors. Run air purifiers with high-efficiency filters.";
    return "Atmospheric emergency. Restrict all outdoor physical activity. Ensure clean-air environments.";
  };

  const activeStationName = STATIONS.find(s => s.id === selectedStation)?.name || "Sector-62, Noida";

  // Trigger Gemini AI forecasting report
  const handleGenerateAiReport = async () => {
    setAiLoading(true);
    try {
      const response = await fetch("/api/gemini/explain-prediction", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pollutants,
          meteorology,
          station: activeStationName,
          predictedAqi,
          forecastHorizon,
          modelName: activeModel
        })
      });
      const data = await response.json();
      setAiReport(data);
    } catch (err) {
      console.error(err);
    } finally {
      setAiLoading(false);
    }
  };

  // Pre-fill sliders with severe smog presets
  const applyPreset = (preset: "winter" | "monsoon" | "clean") => {
    if (preset === "winter") {
      setPollutants({ pm25: 220, pm10: 340, no2: 68, so2: 12, co: 2.2, o3: 25, nh3: 15 });
      setMeteorology({ temperature: 14, humidity: 85, windSpeed: 3.5, rainfall: 0 });
    } else if (preset === "monsoon") {
      setPollutants({ pm25: 35, pm10: 72, no2: 18, so2: 4, co: 0.6, o3: 20, nh3: 6 });
      setMeteorology({ temperature: 29, humidity: 92, windSpeed: 12.5, rainfall: 24 });
    } else {
      setPollutants({ pm25: 15, pm10: 35, no2: 10, so2: 2, co: 0.3, o3: 15, nh3: 4 });
      setMeteorology({ temperature: 22, humidity: 45, windSpeed: 16.0, rainfall: 0 });
    }
  };

  // Helper to get Category colors
  const getCategoryTheme = (aqi: number) => {
    if (aqi <= 50) return { label: "Good", color: "text-emerald-400", border: "border-emerald-500/20", bg: "bg-emerald-500/10", barColor: "#10b981" };
    if (aqi <= 100) return { label: "Satisfactory", color: "text-green-400", border: "border-green-500/20", bg: "bg-green-500/10", barColor: "#22c55e" };
    if (aqi <= 200) return { label: "Moderate", color: "text-yellow-400", border: "border-yellow-500/20", bg: "bg-yellow-500/10", barColor: "#eab308" };
    if (aqi <= 300) return { label: "Poor", color: "text-orange-400", border: "border-orange-500/20", bg: "bg-orange-500/10", barColor: "#f97316" };
    if (aqi <= 400) return { label: "Very Poor", color: "text-red-400", border: "border-red-500/20", bg: "bg-red-500/10", barColor: "#ef4444" };
    return { label: "Severe", color: "text-purple-400", border: "border-purple-500/20", bg: "bg-purple-500/10", barColor: "#a855f7" };
  };

  const catTheme = getCategoryTheme(predictedAqi);

  const allPoints = [...historical, ...forecast];
  const maxVal = Math.max(...allPoints, 350) + 50;

  // Map indices to coordinate pairs on a 500x140 grid
  const getCoords = () => {
    const totalPoints = allPoints.length;
    const widthStep = 480 / (totalPoints - 1);
    return allPoints.map((val, idx) => {
      const x = 10 + idx * widthStep;
      const y = 130 - (val / maxVal) * 110;
      return { x, y, isForecast: idx >= historical.length };
    });
  };

  const coords = getCoords();
  const linePath = coords.map((c, idx) => `${idx === 0 ? "M" : "L"} ${c.x} ${c.y}`).join(" ");

  // Create confidence band translucent polygon coordinates
  const confidencePolygon = () => {
    const hLen = historical.length;
    const widthStep = 480 / (allPoints.length - 1);
    
    // Top line of polygon going forward, then bottom line going backward
    const topPoints = [];
    const bottomPoints = [];

    coords.forEach((c, idx) => {
      if (idx < hLen) {
        // No variance for historical measurements
        topPoints.push(`${c.x},${c.y}`);
        bottomPoints.unshift(`${c.x},${c.y}`);
      } else {
        const variance = (idx - hLen + 1) * 12; // growing uncertainty
        topPoints.push(`${c.x},${Math.max(5, c.y - variance)}`);
        bottomPoints.unshift(`${c.x},${Math.min(135, c.y + variance)}`);
      }
    });

    return [...topPoints, ...bottomPoints].join(" ");
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-widest font-semibold">Simulation sandbox</span>
          <h2 className="text-xl font-display font-bold text-white mt-1">Live Multi-Variable AQI Prognosis</h2>
          <p className="text-xs text-slate-400">Tweak environmental covariates and pollutants to trace real-time prediction responses and confidence cones.</p>
        </div>

        {/* Presets and Horizon selection */}
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={() => applyPreset("winter")} className="px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 hover:bg-red-500/20 text-red-400 font-mono text-xs cursor-pointer transition">
            Winter Preset
          </button>
          <button onClick={() => applyPreset("monsoon")} className="px-3 py-1.5 rounded-lg bg-blue-500/10 border border-blue-500/20 hover:bg-blue-500/20 text-blue-400 font-mono text-xs cursor-pointer transition">
            Monsoon Preset
          </button>
          <button onClick={() => applyPreset("clean")} className="px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 hover:bg-emerald-500/20 text-emerald-400 font-mono text-xs cursor-pointer transition">
            Clean Air Preset
          </button>
        </div>
      </div>

      {/* Target Context Selectors (Station, Date, Time, Model) */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4.5 rounded-2xl bg-white/[0.03] border border-white/10 backdrop-blur-md">
        <div className="space-y-1.5">
          <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block font-semibold">Target Monitoring Station</label>
          <select
            value={selectedStation}
            onChange={(e) => setSelectedStation(e.target.value)}
            className="w-full bg-[#0A0E14] text-white border border-white/10 rounded-lg px-3 py-2 text-xs font-mono focus:border-cyan-400 outline-none"
          >
            {STATIONS.map((s) => (
              <option key={s.id} value={s.id}>{s.name}</option>
            ))}
          </select>
        </div>

        <div className="space-y-1.5">
          <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block font-semibold">Simulation Forecast Date</label>
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="w-full bg-[#0A0E14] text-white border border-white/10 rounded-lg px-3 py-2 text-xs font-mono focus:border-cyan-400 outline-none"
          />
        </div>

        <div className="space-y-1.5">
          <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block font-semibold">Simulation Forecast Time</label>
          <input
            type="time"
            value={selectedTime}
            onChange={(e) => setSelectedTime(e.target.value)}
            className="w-full bg-[#0A0E14] text-white border border-white/10 rounded-lg px-3 py-2 text-xs font-mono focus:border-cyan-400 outline-none"
          />
        </div>

        <div className="space-y-1.5">
          <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block font-semibold">Predictive ML Model</label>
          <select
            value={activeModel}
            onChange={(e) => setActiveModel(e.target.value)}
            className="w-full bg-[#0A0E14] text-white border border-white/10 rounded-lg px-3 py-2 text-xs font-mono focus:border-cyan-400 outline-none"
          >
            <option value="HistGradientBoosting">HistGradientBoostingRegressor (Best)</option>
            <option value="LightGBM">LightGBM</option>
            <option value="XGBoost">XGBoost</option>
            <option value="RandomForest">Random Forest</option>
            <option value="CNN-LSTM">CNN-LSTM Hybrid</option>
            <option value="LSTM">LSTM Sequence Model</option>
            <option value="GRU">GRU Sequence Model</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Side: Hyper-parameter Sliders */}
        <div className="col-span-1 lg:col-span-8 bg-white/[0.03] border border-white/10 rounded-2xl p-5 space-y-6 backdrop-blur-md">
          
          {/* Section: Air Pollutants */}
          <div>
            <div className="flex items-center justify-between mb-4 pb-2 border-b border-white/10">
              <h3 className="text-xs font-mono uppercase tracking-widest text-white/80 font-semibold flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-cyan-400" />
                Air Pollutant Covariates (µg/m³)
              </h3>
              <span className="text-[10px] text-slate-500 font-mono">Input parameters</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
              {/* PM2.5 */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">PM2.5 (Fine Particles)</span>
                  <span className="text-cyan-400 font-semibold">{pollutants.pm25}</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="350"
                  value={pollutants.pm25}
                  onChange={(e) => setPollutants({ ...pollutants, pm25: Number(e.target.value) })}
                  className="w-full accent-cyan-500 cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
                />
              </div>

              {/* PM10 */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">PM10 (Coarse Particles)</span>
                  <span className="text-cyan-400 font-semibold">{pollutants.pm10}</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="500"
                  value={pollutants.pm10}
                  onChange={(e) => setPollutants({ ...pollutants, pm10: Number(e.target.value) })}
                  className="w-full accent-cyan-500 cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
                />
              </div>

              {/* NO2 */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">Nitrogen Dioxide (NO2)</span>
                  <span className="text-cyan-400 font-semibold">{pollutants.no2}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="150"
                  value={pollutants.no2}
                  onChange={(e) => setPollutants({ ...pollutants, no2: Number(e.target.value) })}
                  className="w-full accent-cyan-500 cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
                />
              </div>

              {/* CO */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">Carbon Monoxide (CO - ppm)</span>
                  <span className="text-cyan-400 font-semibold">{pollutants.co}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="8.0"
                  step="0.1"
                  value={pollutants.co}
                  onChange={(e) => setPollutants({ ...pollutants, co: Number(e.target.value) })}
                  className="w-full accent-cyan-500 cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
                />
              </div>

              {/* Ozone */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">Photochemical Ozone (O3)</span>
                  <span className="text-cyan-400 font-semibold">{pollutants.o3}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="200"
                  value={pollutants.o3}
                  onChange={(e) => setPollutants({ ...pollutants, o3: Number(e.target.value) })}
                  className="w-full accent-cyan-500 cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
                />
              </div>

              {/* SO2 */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">Sulfur Dioxide (SO2)</span>
                  <span className="text-cyan-400 font-semibold">{pollutants.so2}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="80"
                  value={pollutants.so2}
                  onChange={(e) => setPollutants({ ...pollutants, so2: Number(e.target.value) })}
                  className="w-full accent-cyan-500 cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
                />
              </div>
            </div>
          </div>

          {/* Section: Meteorological Forcings */}
          <div>
            <div className="flex items-center justify-between mb-4 pb-2 border-b border-white/10">
              <h3 className="text-xs font-mono uppercase tracking-widest text-white/80 font-semibold flex items-center gap-1.5">
                <Wind className="w-4 h-4 text-cyan-400" />
                Meteorological Boundary Forcings
              </h3>
              <span className="text-[10px] text-slate-500 font-mono">Dispersion covariates</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
              {/* Wind Speed */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="flex items-center gap-1 text-slate-400"><Wind className="w-3.5 h-3.5 text-cyan-400" /> Wind Velocity (km/h)</span>
                  <span className="text-cyan-400 font-semibold">{meteorology.windSpeed}</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="35.0"
                  step="0.5"
                  value={meteorology.windSpeed}
                  onChange={(e) => setMeteorology({ ...meteorology, windSpeed: Number(e.target.value) })}
                  className="w-full accent-cyan-500 cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
                />
              </div>

              {/* Humidity */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="flex items-center gap-1 text-slate-400"><Droplets className="w-3.5 h-3.5 text-indigo-400" /> Relative Humidity (%)</span>
                  <span className="text-cyan-400 font-semibold">{meteorology.humidity}%</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="100"
                  value={meteorology.humidity}
                  onChange={(e) => setMeteorology({ ...meteorology, humidity: Number(e.target.value) })}
                  className="w-full accent-cyan-500 cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
                />
              </div>

              {/* Temperature */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="flex items-center gap-1 text-slate-400"><Thermometer className="w-3.5 h-3.5 text-orange-400" /> Ambient Temperature (°C)</span>
                  <span className="text-cyan-400 font-semibold">{meteorology.temperature}°C</span>
                </div>
                <input
                  type="range"
                  min="4"
                  max="45"
                  value={meteorology.temperature}
                  onChange={(e) => setMeteorology({ ...meteorology, temperature: Number(e.target.value) })}
                  className="w-full accent-cyan-500 cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
                />
              </div>

              {/* Rainfall */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-mono">
                  <span className="flex items-center gap-1 text-slate-400"><CloudRain className="w-3.5 h-3.5 text-blue-400" /> Wet Scavenging / Rainfall (mm)</span>
                  <span className="text-cyan-400 font-semibold">{meteorology.rainfall} mm</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="50"
                  value={meteorology.rainfall}
                  onChange={(e) => setMeteorology({ ...meteorology, rainfall: Number(e.target.value) })}
                  className="w-full accent-cyan-500 cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
                />
              </div>
            </div>
          </div>

          {/* Model Param Controls */}
          <div className="p-4 rounded-xl bg-[#0A0E14] border border-white/10 flex flex-wrap gap-4 justify-between items-center">
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-xs font-mono text-slate-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={sensorDrift}
                  onChange={(e) => setSensorDrift(e.target.checked)}
                  className="accent-cyan-500"
                />
                Simulate Sensor drift (+25 PM2.5 drift)
              </label>
              <label className="flex items-center gap-2 text-xs font-mono text-slate-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={confidenceInterval}
                  onChange={(e) => setConfidenceInterval(e.target.checked)}
                  className="accent-cyan-500"
                />
                Show 95% confidence cone
              </label>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-slate-400">Forecast Horizon:</span>
              <select
                value={forecastHorizon}
                onChange={(e) => setForecastHorizon(Number(e.target.value))}
                className="bg-[#0A0E14] text-white border border-white/10 rounded px-2.5 py-1 text-xs font-mono"
              >
                <option value={12}>12 Hours</option>
                <option value={24}>24 Hours</option>
                <option value={48}>48 Hours</option>
                <option value={72}>72 Hours</option>
              </select>
            </div>
          </div>

          {/* Prediction History Log */}
          <div className="p-5 rounded-2xl bg-[#0A0E14] border border-white/10 space-y-3.5 mt-6">
            <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-white/80 flex items-center gap-1.5 pb-2 border-b border-white/10">
              <RefreshCw className="w-4 h-4 text-cyan-400" />
              Interactive Prediction History Log
            </h4>
            <div className="overflow-x-auto">
              <table className="w-full text-[11px] font-mono text-left">
                <thead>
                  <tr className="text-slate-500 border-b border-white/5 pb-1">
                    <th className="py-2 font-normal uppercase text-[9px] tracking-wider text-white/40">Timestamp</th>
                    <th className="py-2 font-normal uppercase text-[9px] tracking-wider text-white/40">Station</th>
                    <th className="py-2 font-normal uppercase text-[9px] tracking-wider text-white/40">Model</th>
                    <th className="py-2 font-normal uppercase text-[9px] tracking-wider text-white/40 text-center">Predicted AQI</th>
                    <th className="py-2 font-normal uppercase text-[9px] tracking-wider text-white/40 text-right">Category</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {predictionHistory.map((item) => (
                    <tr key={item.id} className="hover:bg-white/[0.02]">
                      <td className="py-2 text-slate-400">{item.timestamp}</td>
                      <td className="py-2 text-slate-200">{item.station}</td>
                      <td className="py-2 text-cyan-400">{item.model}</td>
                      <td className="py-2 text-center text-white font-bold">{item.predictedAqi}</td>
                      <td className={`py-2 text-right font-bold ${getCategoryTheme(item.predictedAqi).color}`}>{item.category}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

        </div>

        {/* Right Side: Predicted Output & AI Explainer */}
        <div className="col-span-1 lg:col-span-4 flex flex-col gap-6">
          
          {/* Output Card */}
          <div className="bg-[#0A0E14] border border-white/10 rounded-2xl p-5 flex flex-col justify-between shadow-[0_0_20px_rgba(6,182,212,0.1)]">
            <div>
              <span className="text-[10px] font-mono uppercase text-white/40 block tracking-widest">
                TEMPORAL INFERENCE OUTCOME
              </span>
              <div className="flex items-baseline gap-2.5 mt-2">
                <span className="text-4xl font-display font-bold text-white">{predictedAqi}</span>
                <span className={`text-xs font-mono font-bold uppercase ${catTheme.color}`}>{catTheme.label}</span>
              </div>
              <p className="text-[11px] text-slate-500 font-mono mt-1 leading-relaxed">
                Primary Driver: PM2.5. Forecasted over {forecastHorizon} hours at {activeStationName}.
              </p>
            </div>

            {/* Health Advisory */}
            <div className="mt-4 p-3 rounded-xl bg-cyan-500/5 border border-cyan-500/20 flex gap-2.5 items-start">
              <AlertCircle className={`w-4 h-4 mt-0.5 shrink-0 ${catTheme.color}`} />
              <div className="space-y-0.5">
                <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider block font-semibold">Health Advisory Warning</span>
                <p className="text-[10.5px] text-slate-400 leading-normal font-sans">
                  {getHealthAdvisory(predictedAqi)}
                </p>
              </div>
            </div>

            {/* Custom SVG Line Chart */}
            <div className="mt-4 bg-[#05070A]/80 border border-white/10 p-3 rounded-xl">
              <div className="flex justify-between text-[9px] text-slate-500 font-mono mb-2">
                <span>PAST 24H</span>
                <span>FUTURE {forecastHorizon}H FORECAST</span>
              </div>

              <div className="relative w-full h-32">
                <svg className="w-full h-full overflow-visible">
                  {/* Confidence Interval Band */}
                  {confidenceInterval && (
                    <polygon
                      points={confidencePolygon()}
                      fill="rgba(6, 182, 212, 0.08)"
                      stroke="none"
                    />
                  )}

                  {/* Horizontal Guide lines */}
                  <line x1="10" y1="20" x2="490" y2="20" stroke="rgba(255,255,255,0.04)" strokeDasharray="2,2" />
                  <line x1="10" y1="75" x2="490" y2="75" stroke="rgba(255,255,255,0.04)" strokeDasharray="2,2" />
                  <line x1="10" y1="130" x2="490" y2="130" stroke="rgba(255,255,255,0.04)" strokeDasharray="2,2" />

                  {/* Vertical demarcation */}
                  <line x1="240" y1="5" x2="240" y2="135" stroke="rgba(6,182,212,0.15)" strokeDasharray="3,3" />

                  {/* Trend Path */}
                  <path
                    d={linePath}
                    fill="none"
                    stroke="url(#chart-gradient)"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />

                  {/* Data Point Dots */}
                  {coords.map((c, idx) => (
                    <circle
                      key={idx}
                      cx={c.x}
                      cy={c.y}
                      r="3.5"
                      fill={c.isForecast ? catTheme.barColor : "#94a3b8"}
                      stroke="#020617"
                      strokeWidth="1.5"
                    />
                  ))}

                  {/* Definitions */}
                  <defs>
                    <linearGradient id="chart-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#94a3b8" />
                      <stop offset="50%" stopColor="#22c55e" />
                      <stop offset="100%" stopColor={catTheme.barColor} />
                    </linearGradient>
                  </defs>
                </svg>
              </div>

              <div className="flex justify-between text-[9px] text-slate-500 font-mono mt-1">
                <span>Lag -24h</span>
                <span className="text-cyan-400 font-semibold">T-0 (Now)</span>
                <span>+{forecastHorizon}h</span>
              </div>
            </div>
          </div>

          {/* AI Explainability Prompt Card */}
          <div className="bg-gradient-to-br from-[#12101F]/80 to-[#0A0E14] border border-indigo-500/30 rounded-2xl p-5 space-y-4 shadow-[0_0_20px_rgba(99,102,241,0.15)]">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                <Sparkles className="w-4 h-4 animate-pulse" />
              </div>
              <h4 className="text-xs font-mono font-bold uppercase text-white">AI Forecasting interpretability</h4>
            </div>

            <p className="text-[11px] text-slate-400 leading-relaxed font-sans">
              Request the cloud-deployed **Gemini-3.5-Flash** model to construct a research-grade atmospheric analysis explaining the chemical/physical boundary dynamics of this simulated scenario.
            </p>

            <button
              onClick={handleGenerateAiReport}
              disabled={aiLoading}
              className="w-full flex items-center justify-center gap-2 py-2 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 text-white font-mono text-xs font-bold uppercase shadow-[0_0_15px_rgba(99,102,241,0.4)] transition duration-200 disabled:opacity-50 cursor-pointer"
            >
              {aiLoading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  Synthesizing report...
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  Generate AI Scientific Report
                </>
              )}
            </button>

            {/* AI Generated Output Panel */}
            {aiReport && (
              <div className="p-3.5 rounded-xl bg-[#05070A]/90 border border-indigo-500/35 text-left space-y-2.5 text-xs animate-fade-in font-sans">
                <div>
                  <strong className="text-indigo-400 font-mono block text-[10px] uppercase">FORECAST SUMMARY</strong>
                  <p className="text-slate-200 leading-relaxed mt-0.5">{aiReport.summary}</p>
                </div>
                <div className="border-t border-white/10 pt-2">
                  <strong className="text-indigo-400 font-mono block text-[10px] uppercase">ATMOSPHERIC REASONING</strong>
                  <p className="text-slate-300 leading-relaxed mt-0.5">{aiReport.analysis}</p>
                </div>
                <div className="border-t border-white/10 pt-2">
                  <strong className="text-indigo-400 font-mono block text-[10px] uppercase">HEALTH RECOMMENDATION</strong>
                  <p className="text-slate-200 leading-relaxed mt-0.5">{aiReport.recommendation}</p>
                </div>
                <div className="flex justify-between items-center pt-1 text-[9px] font-mono text-slate-500">
                  <span>Engine: {aiReport.source}</span>
                </div>
              </div>
            )}
          </div>

        </div>

      </div>
    </div>
  );
}
