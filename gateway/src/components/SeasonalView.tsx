import React, { useState, useEffect } from "react";
import { MONTHLY_AQI_PATTERN, generateCalendarData } from "../data";
import { Calendar, Sliders, Info, InfoIcon, ShieldCheck } from "lucide-react";

export default function SeasonalView() {
  const [selectedYear, setSelectedYear] = useState(2026);
  const [hoveredDay, setHoveredDay] = useState<{ date: string; aqi: number } | null>(null);
  const [calendarData, setCalendarData] = useState<any[]>([]);
  const [monthlyPattern, setMonthlyPattern] = useState<any[]>(MONTHLY_AQI_PATTERN);
  const [loading, setLoading] = useState(false);

  // Sync seasonal data from backend on year change
  useEffect(() => {
    let active = true;
    async function fetchTrends() {
      setLoading(true);
      try {
        const res = await fetch(`/api/historical-trends?year=${selectedYear}`);
        const data = await res.json();
        if (active && data.calendarData) {
          setCalendarData(data.calendarData);
          setMonthlyPattern(data.monthlyPattern);
        }
      } catch (err) {
        console.warn("Trends API offline. Reverting to high-fidelity local generator.", err);
        const fallbackData = generateCalendarData(selectedYear);
        if (active) {
          setCalendarData(fallbackData);
          setMonthlyPattern(MONTHLY_AQI_PATTERN);
        }
      } finally {
        if (active) setLoading(false);
      }
    }
    fetchTrends();
    return () => { active = false; };
  }, [selectedYear]);

  // Group days into columns representing weeks (each week has 7 slots)
  const groupIntoWeeks = () => {
    if (!calendarData || calendarData.length === 0) return [];
    const weeks: any[][] = [];
    let currentWeek: any[] = [];
    
    // Pre-fill empty slots for the first week depending on starting day
    const firstDayOfWeek = calendarData[0].dayOfWeek;
    for (let i = 0; i < firstDayOfWeek; i++) {
      currentWeek.push(null);
    }

    calendarData.forEach((day) => {
      currentWeek.push(day);
      if (currentWeek.length === 7) {
        weeks.push(currentWeek);
        currentWeek = [];
      }
    });

    if (currentWeek.length > 0) {
      while (currentWeek.length < 7) {
        currentWeek.push(null);
      }
      weeks.push(currentWeek);
    }
    return weeks;
  };

  const weeks = groupIntoWeeks();

  // Helper for calendar cell coloring
  const getCellBg = (aqi: number) => {
    if (aqi <= 50) return "bg-emerald-950/40 border border-emerald-900/40"; // optimal
    if (aqi <= 100) return "bg-emerald-800/40 border border-emerald-700/40";
    if (aqi <= 150) return "bg-yellow-700/30 border border-yellow-600/30"; // moderate
    if (aqi <= 200) return "bg-amber-600/40 border border-amber-500/40";
    if (aqi <= 300) return "bg-red-800/60 border border-red-700/40"; // poor
    return "bg-purple-900/60 border border-purple-700/40"; // severe smog
  };

  // Helper for tooltip month naming
  const getMonthName = (mIdx: number) => {
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    return months[mIdx];
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-widest font-semibold">Long-term cyclical patterns</span>
          <h2 className="text-xl font-display font-bold text-white mt-1">Multi-Year Seasonal smog Analysis</h2>
          <p className="text-xs text-slate-400">Trace cyclical air quality deviations, monsoonal scavenging wash-outs, and heavy winter thermal inversions.</p>
        </div>

        {/* Year Selectors */}
        <div className="flex bg-[#0A0E14] border border-white/10 p-1 rounded-xl">
          {[2024, 2025, 2026].map((y) => (
            <button
              key={y}
              onClick={() => setSelectedYear(y)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition cursor-pointer ${
                selectedYear === y 
                  ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/30" 
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {y} Calendar
            </button>
          ))}
        </div>
      </div>

      {/* Annual Calendar Heatgrid Grid */}
      <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-5 relative backdrop-blur-md">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs font-mono uppercase tracking-widest text-white/80 font-semibold flex items-center gap-1.5">
            <Calendar className="w-4 h-4 text-cyan-400" />
            Noida Annual AQI violation Heatgrid
          </h3>
          <span className="text-[10px] text-white/40 font-mono">Hover squares to inspect precise AQI logs</span>
        </div>

        {/* Interactive Calendar Map Container */}
        <div className="relative">
          <div className="overflow-x-auto pb-2">
            <div className="flex min-w-[720px] gap-1">
              
              {/* Day Labels */}
              <div className="grid grid-rows-7 text-[8px] font-mono font-semibold text-slate-500 pr-1.5 h-28 items-center pt-2 select-none">
                <span>Sun</span>
                <span>Mon</span>
                <span>Tue</span>
                <span>Wed</span>
                <span>Thu</span>
                <span>Fri</span>
                <span>Sat</span>
              </div>

              {/* Weeks grid mapping */}
              <div className="flex gap-1 h-28 pt-2">
                {weeks.map((week, wIdx) => (
                  <div key={wIdx} className="grid grid-rows-7 gap-1">
                    {week.map((day, dIdx) => {
                      if (!day) {
                        return <div key={dIdx} className="w-2.5 h-2.5 bg-transparent" />;
                      }

                      const isHovered = hoveredDay?.date === day.date;

                      return (
                        <div
                          key={day.date}
                          onMouseEnter={() => setHoveredDay({ date: day.date, aqi: day.aqi })}
                          onMouseLeave={() => setHoveredDay(null)}
                          className={`w-2.5 h-2.5 rounded-[1px] transition duration-100 cursor-pointer ${getCellBg(day.aqi)} ${
                            isHovered ? "ring-2 ring-white scale-125 z-10" : ""
                          }`}
                        />
                      );
                    })}
                  </div>
                ))}
              </div>

            </div>
          </div>

          {/* Floating Tooltip Panel */}
          {hoveredDay && (
            <div className="absolute top-[-25px] left-1/2 transform -translate-x-1/2 p-2 bg-[#0A0E14] border border-white/10 rounded-lg text-[10px] font-mono text-slate-200 shadow-xl flex items-center gap-2 z-20">
              <span className="w-2 h-2 rounded bg-cyan-400" />
              <span>{hoveredDay.date}</span>
              <span className="text-cyan-400 font-bold">AQI {hoveredDay.aqi}</span>
            </div>
          )}
        </div>

        {/* Heatmap Legend */}
        <div className="flex flex-wrap items-center justify-between gap-4 mt-6 pt-3 border-t border-white/10 text-[10px] font-mono text-white/30">
          <div className="flex items-center gap-2.5">
            <span>Cleaner (0-50)</span>
            <div className="flex gap-1">
              <span className="w-2.5 h-2.5 rounded bg-emerald-950/40 border border-emerald-900/40" />
              <span className="w-2.5 h-2.5 rounded bg-emerald-800/40 border border-emerald-700/40" />
              <span className="w-2.5 h-2.5 rounded bg-yellow-700/30 border border-yellow-600/30" />
              <span className="w-2.5 h-2.5 rounded bg-amber-600/40 border border-amber-500/40" />
              <span className="w-2.5 h-2.5 rounded bg-red-800/60 border border-red-700/40" />
              <span className="w-2.5 h-2.5 rounded bg-purple-900/60 border border-purple-700/40" />
            </div>
            <span>Violated (300+)</span>
          </div>
          <span>Reflects winter crop-residual smoke & thermal stagnation events</span>
        </div>
      </div>

      {/* Seasonal Curve Charting Row */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
        
        {/* Monthly Curve Visualizer */}
        <div className="col-span-1 md:col-span-8 bg-white/[0.03] border border-white/10 rounded-2xl p-5 flex flex-col justify-between backdrop-blur-md">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-mono uppercase tracking-widest text-white/80 font-semibold flex items-center gap-1.5">
                <InfoIcon className="w-4 h-4 text-cyan-400" />
                Noida Annual Pollutant Cycle Curve
              </h3>
              <span className="text-[10px] text-white/40 font-mono">Monthly aggregates (2024-2026)</span>
            </div>

            {/* Monthly Curve Chart */}
            <div className="p-3 bg-[#0A0E14] border border-white/10 rounded-xl">
              <div className="relative w-full h-36">
                <svg className="w-full h-full overflow-visible">
                  {/* Vertical demarcation grid */}
                  {monthlyPattern.map((m, idx) => {
                    const x = 20 + idx * 42;
                    return (
                      <line
                        key={idx}
                        x1={x}
                        y1="5"
                        x2={x}
                        y2="135"
                        stroke="rgba(255,255,255,0.06)"
                        strokeWidth="1"
                      />
                    );
                  })}

                  {/* Draw PM2.5 Monthly Line (Red) */}
                  <path
                    d={monthlyPattern.map((m, idx) => {
                      const x = 20 + idx * 42;
                      const y = 135 - (m.pm25 / 260) * 120;
                      return `${idx === 0 ? "M" : "L"} ${x} ${y}`;
                    }).join(" ")}
                    fill="none"
                    stroke="#ef4444"
                    strokeWidth="1.5"
                  />

                  {/* Draw Ozone Monthly Line (Cyan) */}
                  <path
                    d={monthlyPattern.map((m, idx) => {
                      const x = 20 + idx * 42;
                      const y = 135 - (m.o3 / 260) * 120;
                      return `${idx === 0 ? "M" : "L"} ${x} ${y}`;
                    }).join(" ")}
                    fill="none"
                    stroke="#06b6d4"
                    strokeWidth="1.5"
                  />

                  {/* Month Dots and Labels */}
                  {monthlyPattern.map((m, idx) => {
                    const x = 20 + idx * 42;
                    const pmY = 135 - (m.pm25 / 260) * 120;
                    return (
                      <g key={idx}>
                        <circle cx={x} cy={pmY} r="3" fill="#ef4444" />
                        <text x={x} y="146" fill="#94a3b8" fontSize="8" textAnchor="middle" fontFamily="monospace">{m.month}</text>
                      </g>
                    );
                  })}
                </svg>
              </div>

              <div className="flex justify-center gap-6 text-[10px] font-mono text-slate-500 mt-4">
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-0.5 bg-[#ef4444] block" /> Fine PM2.5 (Severe Winter Peaks)</span>
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-0.5 bg-[#06b6d4] block" /> Ozone O3 (Photochemical Summer Peaks)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Informative seasonal card */}
        <div className="col-span-1 md:col-span-4 flex flex-col gap-4">
          <div className="p-5 rounded-2xl bg-white/[0.03] border border-white/10 space-y-4 flex flex-col justify-between h-full backdrop-blur-md">
            <div className="space-y-3.5">
              <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200 flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-cyan-400" />
                Inter-seasonal Cycle Description
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed font-sans">
                The annual cycle is divided into three distinct phases in the Noida airshed:
              </p>
              <ul className="space-y-2 text-[11px] font-sans text-slate-300">
                <li>• <strong className="text-red-400 font-mono">Winter (Nov - Feb):</strong> Thermal inversion layers drop to ~50m, capping particulate emissions and causing severe particulate trapping.</li>
                <li>• <strong className="text-amber-400 font-mono">Summer (Apr - Jun):</strong> Extreme temperatures and high UV indices accelerate VOC titration, causing high photochemical Ozone (O3) production.</li>
                <li>• <strong className="text-emerald-400 font-mono">Monsoon (Jul - Sep):</strong> Heavy rainfall provides wet scavenging and aerosol wash-out, cleaning Noida's particulate grids completely.</li>
              </ul>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
