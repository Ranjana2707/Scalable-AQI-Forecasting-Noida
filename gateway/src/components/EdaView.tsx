import React, { useState, useEffect } from "react";
import { CORRELATION_MATRIX } from "../data";
import { LineChart, BarChart2, ShieldCheck, Activity, AlertTriangle, ChevronRight, SlidersHorizontal, Info } from "lucide-react";

export default function EdaView() {
  const [selectedCell, setSelectedCell] = useState<{ r: number; c: number } | null>({ r: 0, c: 6 }); // default to PM2.5 vs WindSpeed
  const [activeDistribution, setActiveDistribution] = useState<"pm25" | "pm10" | "no2" | "o3">("pm25");
  const [correlationMatrix, setCorrelationMatrix] = useState<{ features: string[]; matrix: number[][] }>(CORRELATION_MATRIX);

  // Sync Pearson correlation matrix from backend
  useEffect(() => {
    let active = true;
    async function fetchCorrelations() {
      try {
        const res = await fetch("/api/eda/correlations");
        const data = await res.json();
        if (active && data) {
          setCorrelationMatrix(data);
        }
      } catch (err) {
        console.warn("Correlations API offline. Reverting to local static calibration models.", err);
      }
    }
    fetchCorrelations();
    return () => { active = false; };
  }, []);

  // Detailed textual explanations of chemical/physical correlations
  const getCorrelationNarrative = (f1: string, f2: string, val: number) => {
    if (f1 === f2) return "Self-correlation (1.0). Perfectly collinear baseline.";
    if (f1 === "PM2.5" && f2 === "WindSpeed") return "Strong negative correlation (-0.52). Increased horizontal wind velocities flush fine aerosol particulates out of the lower boundary layer, inhibiting local concentration peaks.";
    if (f1 === "PM2.5" && f2 === "PM10") return "Strong positive correlation (+0.88). Highly collinear aerosol dispersion. Both indicators rise together, driven by massive regional winter agricultural burning and local road dust.";
    if (f1 === "O3" && f2 === "Temp") return "Moderate-high positive correlation (+0.68). Ambient temperature serves as a solar radiation proxy. High temperatures accelerate the photochemical reaction of nitrogen oxides (NOx) and VOCs to form ground-level Ozone.";
    if (f1 === "Humidity" && f2 === "Temp") return "Strong negative correlation (-0.62). High ambient temperatures lower relative humidity in dry autumn/summer cycles, preventing aerosol swelling (hygroscopic growth).";
    if (f1 === "NO2" && f2 === "PM2.5") return "Strong positive correlation (+0.65). Highly representative of vehicular combustion. Noida vehicle emissions release massive amounts of NOx alongside secondary organic aerosols.";
    if (f1 === "O3" && f2 === "NO2") return "Negative correlation (-0.32). Ozone titration. Direct reaction with fresh nitric oxide (NO) emissions destroys ozone locally, leading to lower ozone levels near busy roadways.";
    
    // Generic fallback based on value sign
    const strength = Math.abs(val) > 0.6 ? "Strong" : Math.abs(val) > 0.3 ? "Moderate" : "Weak";
    const direction = val > 0 ? "positive" : "negative";
    return `${strength} ${direction} atmospheric correlation (${val > 0 ? "+" : ""}${val.toFixed(2)}). Reflects typical regional meteorological dispersion patterns within the Indo-Gangetic Plain.`;
  };

  const cellX = selectedCell ? correlationMatrix.features[selectedCell.c] : "";
  const cellY = selectedCell ? correlationMatrix.features[selectedCell.r] : "";
  const cellVal = selectedCell ? correlationMatrix.matrix[selectedCell.r][selectedCell.c] : 0;

  // Let's model a mock sensor calibration log
  const calibrationLogs = [
    { timestamp: "2026-07-10 18:42", parameter: "PM2.5 Sector-62", message: "Zero-point laser calibration completed. Current offset: -1.2 µg/m³", status: "success" },
    { timestamp: "2026-07-10 12:15", parameter: "O3 UV Photometer", message: "Hygroscopic condensation filter purge cycle completed. Flow rate stable (1.2 L/min).", status: "success" },
    { timestamp: "2026-07-10 08:30", parameter: "CO Sector-1 Sensor", message: "Electrothermal drift deviation detected (+0.18 ppm offset). Automatic compensation applied.", status: "warning" },
    { timestamp: "2026-07-09 14:00", parameter: "Telemetry System", message: "Successfully synced with regional CPCB server (GPRS Link). Error rate: 0.02%", status: "success" }
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Top Header */}
      <div>
        <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-widest font-semibold">Scientific exploratory dashboard</span>
        <h2 className="text-xl font-display font-bold text-white mt-1">Boundary Layer Correlation & Data Calibration</h2>
        <p className="text-xs text-slate-400">Examine pollutant interactions, statistical significance grids, and continuous sensor state logs.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Pairwise Correlation Heatmap (8x8) */}
        <div className="col-span-1 lg:col-span-8 bg-white/[0.03] border border-white/10 rounded-2xl p-5 flex flex-col justify-between backdrop-blur-md">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-mono uppercase tracking-widest text-white/80 font-semibold flex items-center gap-1.5">
                <SlidersHorizontal className="w-4 h-4 text-cyan-400" />
                Feature-to-Feature Correlation Heatgrid
              </h3>
              <span className="text-[10px] text-white/40 font-mono">Pearson (R) Matrix</span>
            </div>

            {/* Heatgrid Layout */}
            <div className="overflow-x-auto">
              <div className="min-w-[420px] space-y-1">
                {/* Header Labels */}
                <div className="grid grid-cols-9 text-center text-[10px] font-mono font-bold text-white/30 pb-1.5 border-b border-white/10">
                  <div className="text-left pl-1">Feature</div>
                  {correlationMatrix.features.map((f) => (
                    <div key={f} className="truncate px-0.5">{f}</div>
                  ))}
                </div>

                {/* Heatmap Rows */}
                {correlationMatrix.features.map((rowName, rIdx) => (
                  <div key={rowName} className="grid grid-cols-9 items-center text-center">
                    {/* Y Axis Label */}
                    <div className="text-left text-[10px] font-mono font-semibold text-slate-400 pr-2 truncate">
                      {rowName}
                    </div>

                    {/* Correlation cells */}
                    {correlationMatrix.matrix[rIdx].map((val, cIdx) => {
                      // Positive values -> cyan scale, negative -> coral scale
                      let bgStyle = {};
                      if (val > 0) {
                        bgStyle = { backgroundColor: `rgba(6, 182, 212, ${val * 0.85})` };
                      } else {
                        bgStyle = { backgroundColor: `rgba(239, 68, 68, ${Math.abs(val) * 0.75})` };
                      }

                      const isSelected = selectedCell?.r === rIdx && selectedCell?.c === cIdx;

                      return (
                        <div
                          key={cIdx}
                          onClick={() => setSelectedCell({ r: rIdx, c: cIdx })}
                          style={bgStyle}
                          className={`aspect-square m-0.5 rounded flex items-center justify-center text-[9px] font-mono font-bold cursor-pointer transition-all duration-150 ${
                            val > 0.6 ? "text-slate-950" : "text-slate-100"
                          } ${isSelected ? "ring-2 ring-white scale-105 shadow-lg" : "hover:scale-105 hover:brightness-110"}`}
                        >
                          {val > 0 ? `+${val.toFixed(1)}` : val.toFixed(1)}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Cell Detail Explainer Block */}
          {selectedCell && (
            <div className="mt-6 p-4 rounded-xl bg-[#0A0E14] border border-white/10 flex items-start gap-3.5">
              <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 shrink-0">
                <Info className="w-5 h-5" />
              </div>
              <div className="space-y-1.5 text-xs">
                <div className="flex items-center gap-2">
                  <strong className="text-white font-mono text-xs">{cellY} vs {cellX}</strong>
                  <span className={`text-[10px] font-mono font-bold px-2 py-0.2 rounded ${cellVal > 0 ? "bg-cyan-500/10 text-cyan-400" : "bg-red-500/10 text-red-400"}`}>
                    R = {cellVal > 0 ? "+" : ""}{cellVal.toFixed(2)}
                  </span>
                </div>
                <p className="text-slate-300 leading-relaxed font-sans">{getCorrelationNarrative(cellY, cellX, cellVal)}</p>
              </div>
            </div>
          )}
        </div>

        {/* Right Side: Sensor Logs & Calibration State */}
        <div className="col-span-1 lg:col-span-4 flex flex-col gap-6">
          
          {/* Active Calibration State */}
          <div className="p-5 rounded-2xl bg-[#0A0E14] border border-white/10 flex flex-col justify-between shadow-[0_0_15px_rgba(34,211,238,0.05)]">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-white flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  Sensor Health Profile
                </h4>
                <span className="text-[9px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.2 rounded border border-emerald-500/20 uppercase">PASS</span>
              </div>

              {/* Ingestion Stream Details */}
              <div className="space-y-2 text-xs font-mono">
                <div className="flex justify-between py-1.5 border-b border-white/10">
                  <span className="text-slate-400">Total Noida Sensors</span>
                  <span className="text-slate-200 font-semibold">28 Nodes</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-white/10">
                  <span className="text-slate-400">Active telemetry rate</span>
                  <span className="text-slate-200">1.0 Hz (Every 1s)</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-white/10">
                  <span className="text-slate-400">Zero-drift calibration</span>
                  <span className="text-emerald-400 font-semibold">Automatic</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-slate-400">Relative Humidity drift</span>
                  <span className="text-cyan-400">Compensation Active</span>
                </div>
              </div>
            </div>
          </div>

          {/* Real-time System Anomaly log */}
          <div className="p-5 rounded-2xl bg-white/[0.03] border border-white/10 flex flex-col justify-between h-full backdrop-blur-md">
            <div>
              <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-white flex items-center gap-1.5 mb-3.5">
                <Activity className="w-4 h-4 text-cyan-400" />
                Sensor Calibration Logs
              </h4>

              <div className="space-y-3 max-h-[220px] overflow-y-auto pr-1">
                {calibrationLogs.map((log, idx) => (
                  <div key={idx} className="p-2.5 rounded-lg bg-[#0A0E14]/60 border border-white/10 text-[10px] font-mono space-y-1">
                    <div className="flex justify-between items-center text-white/30">
                      <span>{log.timestamp}</span>
                      <span className={`text-[9px] font-semibold uppercase px-1.5 rounded ${
                        log.status === "warning" ? "bg-amber-500/15 text-amber-400" : "bg-cyan-500/10 text-cyan-400"
                      }`}>
                        {log.status}
                      </span>
                    </div>
                    <div className="text-slate-300 font-semibold">{log.parameter}</div>
                    <p className="text-slate-400 text-[9px] font-sans leading-relaxed">{log.message}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="text-[10px] font-mono text-white/30 text-center mt-3 pt-3 border-t border-white/10">
              System health score: <span className="text-emerald-400 font-semibold">99.8%</span>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
