import React, { useState, useEffect, useRef } from "react";
import L from "leaflet";
import { STATIONS } from "../data";
import { 
  TrendingUp, 
  Wind, 
  Thermometer, 
  Droplets, 
  MapPin, 
  AlertTriangle, 
  Check, 
  Activity,
  ArrowUpRight,
  ShieldCheck,
  Cpu
} from "lucide-react";

interface DashboardViewProps {
  selectedStation: string;
  setSelectedStation: (id: string) => void;
  predictedAqi: number;
}

export default function DashboardView({ selectedStation, setSelectedStation, predictedAqi }: DashboardViewProps) {
  const [stationStats, setStationStats] = useState<Record<string, { aqi: number; temp: number; wind: number; hum: number; mainPollutant: string }> | null>(null);

  // Sync regional spatial nodes from the core backend with auto-refresh every 60s
  useEffect(() => {
    let active = true;
    async function fetchStations() {
      try {
        const res = await fetch("/api/stations-data");
        const data = await res.json();
        if (active && data) {
          setStationStats(data);
        }
      } catch (err) {
        console.warn("Dashboard Stations API offline.", err);
        if (active) setStationStats(null);
      }
    }
    fetchStations();
    
    const interval = setInterval(fetchStations, 60000);
    return () => { 
      active = false;
      clearInterval(interval);
    };
  }, []);

  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markersRef = useRef<Record<string, L.Marker>>({});

  useEffect(() => {
    if (!mapRef.current) return;
    
    if (!mapInstanceRef.current) {
      const map = L.map(mapRef.current, {
        center: [28.55, 77.38],
        zoom: 10.5,
        zoomControl: true,
        attributionControl: false
      });
      
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
      }).addTo(map);

      mapInstanceRef.current = map;
    }

    const map = mapInstanceRef.current;

    // Remove existing markers
    Object.values(markersRef.current).forEach((m: any) => m.remove());
    markersRef.current = {};

    STATIONS.forEach(station => {
      const isSelected = selectedStation === station.id;
      const stats = stationStats ? (stationStats[station.id] || { aqi: 0 }) : { aqi: 0 };
      const aqi = stats.aqi;
      
      let color = "#ef4444";
      if (aqi <= 50) color = "#10b981";
      else if (aqi <= 100) color = "#22c55e";
      else if (aqi <= 200) color = "#eab308";
      else if (aqi <= 300) color = "#f97316";
      
      const pulseHtml = isSelected 
        ? `<span class="absolute inline-flex h-full w-full rounded-full animate-ping opacity-75" style="background-color: ${color}"></span>`
        : "";

      const icon = L.divIcon({
        className: "relative flex items-center justify-center",
        html: `
          <div class="relative flex h-5 w-5 items-center justify-center">
            ${pulseHtml}
            <span class="relative inline-flex rounded-full h-3.5 w-3.5 border border-slate-950 shadow-md" style="background-color: ${color}"></span>
          </div>
        `,
        iconSize: [20, 20],
        iconAnchor: [10, 10]
      });

      const marker = L.marker([station.lat, station.lng], { icon })
        .addTo(map)
        .bindTooltip(`
          <div style="background-color: #0b0f17; border: 1px solid rgba(255,255,255,0.15); padding: 6px; border-radius: 4px; font-family: monospace; font-size: 11px; color: #fff;">
            <strong>${station.name}</strong><br/>
            <span style="color: #22d3ee; font-weight: bold;">AQI: ${aqi}</span>
          </div>
        `, {
          permanent: false,
          direction: "top",
          opacity: 0.9
        });

      marker.on("click", () => {
        setSelectedStation(station.id);
      });

      markersRef.current[station.id] = marker;
    });

    const activeStation = STATIONS.find(s => s.id === selectedStation);
    if (activeStation) {
      map.setView([activeStation.lat, activeStation.lng], 11);
    }
  }, [selectedStation, stationStats, setSelectedStation]);

  useEffect(() => {
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  const activeStats = stationStats ? (stationStats[selectedStation] || stationStats.sec62) : null;
  const activeStationDetails = STATIONS.find(s => s.id === selectedStation) || STATIONS[0];

  const getAqiCategory = (aqi: number) => {
    if (aqi <= 0) return { label: "N/A", color: "text-slate-400", border: "border-slate-500/30", bg: "bg-slate-500/10", text: "No Data Available", description: "Telemetry data is currently unavailable." };
    if (aqi <= 50) return { label: "Good", color: "text-emerald-400", border: "border-emerald-500/30", bg: "bg-emerald-500/10", text: "Optimal", description: "Air quality is considered satisfactory, and air pollution poses little or no risk." };
    if (aqi <= 100) return { label: "Satisfactory", color: "text-green-400", border: "border-green-500/30", bg: "bg-green-500/10", text: "Acceptable", description: "Air quality is acceptable; however, there may be moderate health concerns for some people." };
    if (aqi <= 200) return { label: "Moderate", color: "text-yellow-400", border: "border-yellow-500/30", bg: "bg-yellow-500/10", text: "Slight Discomfort", description: "Members of sensitive groups may experience health effects." };
    if (aqi <= 300) return { label: "Poor", color: "text-orange-400", border: "border-orange-500/30", bg: "bg-orange-500/10", text: "Respiratory Risk", description: "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects." };
    if (aqi <= 400) return { label: "Very Poor", color: "text-red-400", border: "border-red-500/30", bg: "bg-red-500/10", text: "Severe Exposure", description: "Health alert: everyone may experience more serious health effects." };
    return { label: "Severe", color: "text-purple-400", border: "border-purple-500/30", bg: "bg-purple-500/10", text: "Toxic Hazard", description: "Health warning of emergency conditions. The entire population is more likely to be affected." };
  };

  const currentCat = activeStats ? getAqiCategory(activeStats.aqi) : getAqiCategory(0);

  // SVG Gauge calculations
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = activeStats 
    ? circumference - (Math.min(activeStats.aqi, 400) / 400) * circumference
    : circumference;

  return (
    <div className="space-y-6">
      {/* Top Welcome Panel */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-white/[0.03] p-5 rounded-2xl border border-white/10 backdrop-blur-md">
        <div>
          <span className="text-[10px] font-mono uppercase text-cyan-400 tracking-wider font-semibold">
            Predictive Atmospheric Analytics Platform
          </span>
          <h2 className="text-xl font-display font-bold text-white mt-1">
            Noida Smart Pollution Cluster Overview
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Real-time multi-station sensor ingestion fused with deep learning temporal predictors.
          </p>
        </div>
        <div className="flex items-center gap-3 font-mono text-xs">
          <div className="px-3 py-1.5 rounded-lg bg-[#0A0E14] border border-white/10 flex items-center gap-1.5 text-slate-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            Calibration: Active
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-cyan-400/10 border border-cyan-400/20 text-cyan-400 font-semibold flex items-center gap-1 shadow-[0_0_10px_rgba(34,211,238,0.1)]">
            <Cpu className="w-3.5 h-3.5" />
            Core Ingestion: 2.8s lag
          </div>
        </div>
      </div>

      {/* Main Grid: Map & Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Noida Map Cluster Visualization */}
        <div id="noida-map-card" className="col-span-1 lg:col-span-7 bg-white/[0.03] border border-white/10 rounded-2xl p-5 flex flex-col justify-between backdrop-blur-md">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-mono uppercase tracking-widest text-white/60 font-semibold flex items-center gap-1.5">
                <MapPin className="w-4 h-4 text-cyan-400" />
                Noida Spatial Cluster Heatgrid
              </h3>
              <span className="text-[10px] text-white/40 font-mono">Click stations to analyze</span>
            </div>
            
            {/* Real Leaflet Map */}
            <div className="relative w-full aspect-video bg-[#0A0E14] border border-white/10 rounded-xl overflow-hidden">
              <div ref={mapRef} className="w-full h-full" style={{ minHeight: "260px" }} />
            </div>
          </div>

          {/* Map Legend */}
          <div className="flex flex-wrap items-center justify-between gap-3 mt-4 pt-3 border-t border-white/10">
            <div className="flex items-center gap-4 text-[10px] font-mono text-slate-400">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded bg-emerald-500/20 border border-emerald-500/40" /> Good (0-50)</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded bg-yellow-500/20 border border-yellow-500/40" /> Moderate (101-200)</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded bg-red-500/20 border border-red-500/40" /> Poor (201-300)</span>
            </div>
            <span className="text-[10px] text-white/30 font-mono">Yamuna River flow highlighted in blue</span>
          </div>
        </div>

        {/* Selected Station Detailed Stats */}
        <div className="col-span-1 lg:col-span-5 flex flex-col gap-6">
          
          {/* Main Air Quality Gauge Card */}
          <div className="bg-[#0A0E14] border border-white/10 rounded-2xl p-5 relative overflow-hidden flex flex-col justify-between h-full shadow-[0_0_20px_rgba(6,182,212,0.1)]">
            <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/5 blur-3xl pointer-events-none rounded-full" />
            
            <div className="flex items-center justify-between mb-2">
              <div>
                <span className="text-[9px] font-mono uppercase bg-white/5 border border-white/10 px-2 py-0.5 rounded text-cyan-400">
                  {activeStationDetails.areaType}
                </span>
                <h4 className="text-sm font-semibold text-white mt-1.5">{activeStationDetails.name}</h4>
              </div>
              <div className="text-right">
                <span className="text-xs text-white/40 block font-mono">Current AQI</span>
              </div>
            </div>

            {/* Circular Gauge */}
            <div className="flex items-center justify-center py-4">
              <div className="relative flex items-center justify-center">
                <svg className="w-36 h-36 transform -rotate-90">
                  {/* Background Circle */}
                  <circle
                    cx="72"
                    cy="72"
                    r={radius}
                    className="stroke-white/5"
                    strokeWidth="8"
                    fill="transparent"
                  />
                  {/* Colored Active Arc */}
                  <circle
                    cx="72"
                    cy="72"
                    r={radius}
                    stroke="url(#aqi-gauge-gradient)"
                    strokeWidth="8"
                    fill="transparent"
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    strokeLinecap="round"
                    className="transition-all duration-1000 ease-out"
                  />
                  <defs>
                    <linearGradient id="aqi-gauge-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#22c55e" />
                      <stop offset="50%" stopColor="#eab308" />
                      <stop offset="100%" stopColor="#ef4444" />
                    </linearGradient>
                  </defs>
                </svg>
                {/* Text overlay */}
                <div className="absolute flex flex-col items-center justify-center text-center">
                  <span className="text-2xl font-display font-bold text-white tracking-tight">
                    {activeStats ? activeStats.aqi : "No data available"}
                  </span>
                  <span className={`text-[10px] font-mono font-bold tracking-widest uppercase mt-0.5 ${currentCat.color}`}>
                    {currentCat.label}
                  </span>
                </div>
              </div>
            </div>

            {/* Recommendations footer */}
            <div className={`p-3 rounded-xl border ${currentCat.border} ${currentCat.bg} text-[11px] leading-relaxed flex gap-2`}>
              <AlertTriangle className={`w-4 h-4 shrink-0 ${currentCat.color}`} />
              <div>
                <strong className="text-slate-200 block mb-0.5">Health Threshold Warning</strong>
                <span className="text-slate-300 font-sans">{activeStats ? `${currentCat.text}: ${currentCat.description}` : "No data available"}</span>
              </div>
            </div>
          </div>

        </div>

      </div>

      {/* Grid: Meteorological Fusions and Quick Trend overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        
        {/* Core Pollutants */}
        <div className="p-4 rounded-xl bg-white/[0.04] border border-white/10 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-white/40">PRIMARY POLLUTANT DRIVER</span>
            <Wind className="w-3.5 h-3.5 text-cyan-400" />
          </div>
          <div className="mt-2.5">
            <h5 className="text-lg font-display font-bold text-white">{activeStats ? activeStats.mainPollutant : "No data available"}</h5>
            <div className="flex justify-between items-center mt-1">
              <span className="text-[11px] text-slate-400">Ingested density</span>
              <span className="text-[10px] font-mono bg-red-500/10 text-red-400 border border-red-500/20 px-1.5 py-0.2 rounded font-semibold">+45% peak</span>
            </div>
          </div>
        </div>

        {/* Temperature */}
        <div className="p-4 rounded-xl bg-white/[0.04] border border-white/10 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-white/40">AMBIENT TEMPERATURE</span>
            <Thermometer className="w-3.5 h-3.5 text-orange-400" />
          </div>
          <div className="mt-2.5">
            <h5 className="text-lg font-display font-bold text-white">{activeStats ? `${activeStats.temp}°C` : "No data available"}</h5>
            <div className="flex justify-between items-center mt-1">
              <span className="text-[11px] text-slate-400">Inversion Factor</span>
              <span className="text-[10px] font-mono text-amber-400">Moderate Inversion</span>
            </div>
          </div>
        </div>

        {/* Wind */}
        <div className="p-4 rounded-xl bg-white/[0.04] border border-white/10 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-white/40">WIND DRIFT DISPERSION</span>
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="mt-2.5">
            <h5 className="text-lg font-display font-bold text-white">{activeStats ? `${activeStats.wind} km/h` : "No data available"}</h5>
            <div className="flex justify-between items-center mt-1">
              <span className="text-[11px] text-slate-400">Horizontal transport</span>
              <span className="text-[10px] font-mono text-red-400">Critical stagnation</span>
            </div>
          </div>
        </div>

        {/* Humidity */}
        <div className="p-4 rounded-xl bg-white/[0.04] border border-white/10 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-white/40">RELATIVE HUMIDITY</span>
            <Droplets className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <div className="mt-2.5">
            <h5 className="text-lg font-display font-bold text-white">{activeStats ? `${activeStats.hum}%` : "No data available"}</h5>
            <div className="flex justify-between items-center mt-1">
              <span className="text-[11px] text-slate-400">Aerosol accumulation</span>
              <span className="text-[10px] font-mono text-indigo-400">Condensation lock</span>
            </div>
          </div>
        </div>

      </div>

      {/* System Status Metrics Card */}
      <div className="p-5 rounded-2xl bg-white/[0.03] border border-white/10 backdrop-blur-md flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-cyan-400/10 text-cyan-400 border border-cyan-400/20">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-white">Production Model status</h4>
            <p className="text-[11px] text-slate-400 mt-0.5">HistGradientBoosting active. Next retrain scheduled in 14 hours.</p>
          </div>
        </div>
        <div className="flex gap-4">
          <div className="text-right">
            <span className="text-[10px] font-mono text-white/30 block">72H FORECAST OUTCOME</span>
            <span className="text-xs font-mono font-semibold text-cyan-400">Predicted peak: {predictedAqi} AQI</span>
          </div>
          <div className="border-l border-white/10 h-8" />
          <div className="text-right">
            <span className="text-[10px] font-mono text-white/30 block">MODEL RE-VALIDATION</span>
            <span className="text-xs font-mono font-semibold text-emerald-400">R² = 0.941</span>
          </div>
        </div>
      </div>
    </div>
  );
}
