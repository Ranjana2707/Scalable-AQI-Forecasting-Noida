import React, { useState, useEffect } from "react";
import { STATIONS } from "../data";
import { Map, Info, Compass, Shield, Award, MapPin } from "lucide-react";

export default function StationsView() {
  const [selectedNode, setSelectedNode] = useState("sec62");
  const [stationProfiles, setStationProfiles] = useState<Record<string, { pm25: number; pm10: number; no2: number; o3: number; co: number; so2: number; peakHour: string; healthRisk: string }>>({
    sec62: { pm25: 210, pm10: 295, no2: 62, o3: 34, co: 1.8, so2: 8, peakHour: "18:00 - 21:00", healthRisk: "High (Chronic Exposure)" },
    sec125: { pm25: 168, pm10: 220, no2: 45, o3: 42, co: 1.2, so2: 5, peakHour: "08:30 - 10:30", healthRisk: "Moderate (Transitional)" },
    kp3: { pm25: 142, pm10: 160, no2: 24, o3: 48, co: 0.8, so2: 3, peakHour: "13:00 - 15:00", healthRisk: "Slight (Low Threshold)" },
    sec1: { pm25: 184, pm10: 240, no2: 52, o3: 38, co: 1.5, so2: 6, peakHour: "17:30 - 20:00", healthRisk: "High (Commercial Density)" }
  });

  // Sync regional spatial nodes from the core backend
  useEffect(() => {
    let active = true;
    async function fetchStations() {
      try {
        const res = await fetch("/api/stations-data");
        const data = await res.json();
        if (active && data) {
          setStationProfiles(data);
        }
      } catch (err) {
        console.warn("Stations API offline. Reverting to local reference models.", err);
      }
    }
    fetchStations();
    return () => { active = false; };
  }, []);

  const activeNodeDetails = STATIONS.find(s => s.id === selectedNode) || STATIONS[0];
  const activeProfile = stationProfiles[selectedNode] || stationProfiles.sec62;

  // Let's draw a beautiful SVG Radar chart representing the 6 pollutants
  // We'll calculate the 6 coordinates (60-degree increments) on a 200x200 canvas centered at 100, 100.
  const getRadarCoordinates = () => {
    const center = 100;
    // Map max limits to 100px radius
    const scales = {
      pm25: 220,
      pm10: 320,
      no2: 80,
      o3: 60,
      co: 2.5,
      so2: 12
    };

    const vars = [
      { key: "pm25", label: "PM2.5", val: activeProfile.pm25, max: scales.pm25 },
      { key: "pm10", label: "PM10", val: activeProfile.pm10, max: scales.pm10 },
      { key: "no2", label: "NO2", val: activeProfile.no2, max: scales.no2 },
      { key: "o3", label: "O3", val: activeProfile.o3, max: scales.o3 },
      { key: "co", label: "CO", val: activeProfile.co, max: scales.co },
      { key: "so2", label: "SO2", val: activeProfile.so2, max: scales.so2 }
    ];

    return vars.map((v, idx) => {
      const angle = (idx * 60 * Math.PI) / 180;
      const ratio = Math.min(v.val / v.max, 1.0);
      const radius = 10 + ratio * 80; // keep inside bounds
      const x = center + radius * Math.cos(angle - Math.PI / 2);
      const y = center + radius * Math.sin(angle - Math.PI / 2);
      
      // Calculate background polygon points for grid lines
      const gridCoords = [0.25, 0.5, 0.75, 1.0].map((gRatio) => {
        const gRadius = 10 + gRatio * 80;
        return {
          x: center + gRadius * Math.cos(angle - Math.PI / 2),
          y: center + gRadius * Math.sin(angle - Math.PI / 2)
        };
      });

      return { key: v.label, x, y, val: v.val, gridCoords };
    });
  };

  const coords = getRadarCoordinates();
  const filledPath = coords.map((c, idx) => `${idx === 0 ? "M" : "L"} ${c.x} ${c.y}`).join(" ") + " Z";

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-widest font-semibold">Spatial node analysis</span>
        <h2 className="text-xl font-display font-bold text-white mt-1">Noida Multi-Station Pollution Cluster Comparison</h2>
        <p className="text-xs text-slate-400">Examine how unique local industrial operations and vehicular congestion shape multi-pollutant footprints.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Radar Footprint Chart */}
        <div className="col-span-1 lg:col-span-7 bg-white/[0.03] border border-white/10 rounded-2xl p-5 flex flex-col justify-between backdrop-blur-md">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-mono uppercase tracking-widest text-white/80 font-semibold flex items-center gap-1.5">
                <Compass className="w-4 h-4 text-cyan-400" />
                Pollutant Concentration Fingerprint
              </h3>
              <span className="text-[10px] text-white/40 font-mono">Radial foot profile</span>
            </div>

            {/* Radar footprint Canvas */}
            <div className="flex flex-col md:flex-row items-center justify-around gap-6 bg-[#0A0E14] p-4 rounded-xl border border-white/10">
              <div className="relative w-56 h-56">
                <svg className="w-full h-full overflow-visible" viewBox="0 0 200 200">
                  {/* Draw Concentric hexagonal grids */}
                  {[0.25, 0.5, 0.75, 1.0].map((gRatio, gIdx) => {
                    const gridPath = coords.map((c) => {
                      const gc = c.gridCoords[gIdx];
                      return `${gc.x},${gc.y}`;
                    }).join(" ");
                    
                    return (
                      <polygon
                        key={gIdx}
                        points={gridPath}
                        fill="none"
                        stroke="rgba(255,255,255,0.08)"
                        strokeWidth="1"
                      />
                    );
                  })}

                  {/* Draw Spokes */}
                  {coords.map((c, idx) => {
                    const angle = (idx * 60 * Math.PI) / 180;
                    const x2 = 100 + 90 * Math.cos(angle - Math.PI / 2);
                    const y2 = 100 + 90 * Math.sin(angle - Math.PI / 2);
                    return (
                      <line
                        key={idx}
                        x1="100"
                        y1="100"
                        x2={x2}
                        y2={y2}
                        stroke="rgba(255,255,255,0.12)"
                        strokeWidth="1"
                      />
                    );
                  })}

                  {/* Draw Active Fingerprint Polygon */}
                  <path
                    d={filledPath}
                    fill="rgba(6, 182, 212, 0.12)"
                    stroke="rgba(6, 182, 212, 0.6)"
                    strokeWidth="2"
                    className="transition-all duration-300"
                  />

                  {/* Draw Labels on axis tips */}
                  {coords.map((c, idx) => {
                    const angle = (idx * 60 * Math.PI) / 180;
                    const lx = 100 + 104 * Math.cos(angle - Math.PI / 2);
                    const ly = 100 + 104 * Math.sin(angle - Math.PI / 2);
                    return (
                      <text
                        key={idx}
                        x={lx}
                        y={ly + 3}
                        fill="#94a3b8"
                        fontSize="7"
                        fontWeight="bold"
                        textAnchor="middle"
                        fontFamily="monospace"
                      >
                        {c.key}
                      </text>
                    );
                  })}

                  {/* Axis dots */}
                  {coords.map((c, idx) => (
                    <circle
                      key={idx}
                      cx={c.x}
                      cy={c.y}
                      r="3.5"
                      fill="#06b6d4"
                      stroke="#020617"
                      strokeWidth="1.5"
                    />
                  ))}
                </svg>
              </div>

              {/* Dynamic radar footnotes */}
              <div className="space-y-3 flex-1 text-xs font-mono">
                <span className="text-[10px] font-bold text-cyan-400 block tracking-wider">CURRENT STRENGTHS:</span>
                <div className="space-y-1.5">
                  <div className="flex justify-between border-b border-white/10 pb-1">
                    <span className="text-slate-400">PM2.5 Density</span>
                    <span className="text-slate-200 font-semibold">{activeProfile.pm25} µg/m³</span>
                  </div>
                  <div className="flex justify-between border-b border-white/10 pb-1">
                    <span className="text-slate-400">NO2 Level</span>
                    <span className="text-slate-200">{activeProfile.no2} ppb</span>
                  </div>
                  <div className="flex justify-between border-b border-white/10 pb-1">
                    <span className="text-slate-400">O3 Level</span>
                    <span className="text-slate-200">{activeProfile.o3} ppb</span>
                  </div>
                  <div className="flex justify-between pb-1">
                    <span className="text-slate-400">Peak hour</span>
                    <span className="text-amber-400">{activeProfile.peakHour}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Side: Station Selection Grid & Details */}
        <div className="col-span-1 lg:col-span-5 flex flex-col gap-4">
          
          <div className="p-5 rounded-2xl bg-white/[0.03] border border-white/10 flex flex-col justify-between h-full backdrop-blur-md">
            <div>
              <h3 className="text-xs font-mono uppercase tracking-widest text-white/80 font-semibold flex items-center gap-1.5 mb-4">
                <MapPin className="w-4 h-4 text-cyan-400" />
                Select Ingestion Node
              </h3>

              {/* Station grid list */}
              <div className="space-y-2.5">
                {STATIONS.map((s) => {
                  const isSelected = selectedNode === s.id;
                  const stProf = stationProfiles[s.id] || stationProfiles.sec62;
                  
                  return (
                    <div
                      key={s.id}
                      onClick={() => setSelectedNode(s.id)}
                      className={`p-3.5 rounded-xl border cursor-pointer transition duration-150 ${
                        isSelected 
                          ? "bg-cyan-500/10 border-cyan-500/40 shadow-[0_0_12px_rgba(6,182,212,0.15)]" 
                          : "bg-[#0A0E14]/50 border-white/5 hover:bg-[#0A0E14]/90"
                      }`}
                    >
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs font-semibold text-slate-200">{s.name}</span>
                        <span className={`text-[10px] font-mono font-bold uppercase ${isSelected ? "text-cyan-400" : "text-slate-500"}`}>
                          AQI: {stProf.pm25}
                        </span>
                      </div>
                      <div className="flex justify-between text-[10px] font-mono text-slate-500 leading-none">
                        <span>{s.areaType}</span>
                        <span>{s.lat.toFixed(4)}N, {s.lng.toFixed(4)}E</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-[#0A0E14] border border-white/10 text-xs text-slate-400 mt-4 leading-relaxed font-sans">
              <strong>Atmospheric Footprint Description:</strong> {activeNodeDetails.areaType} zone profile showing typical {activeProfile.peakHour} peaks driven by local variables.
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
