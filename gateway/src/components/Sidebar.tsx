import React from "react";
import {
  LayoutDashboard,
  LineChart,
  Sliders,
  Eye,
  Cpu,
  Layers,
  Map,
  Calendar,
  BookOpen,
  Activity,
  Sparkles
} from "lucide-react";
import { Tab } from "../types";

interface SidebarProps {
  activeTab: Tab;
  setActiveTab: (tab: Tab) => void;
  aiConnected: boolean;
}

export default function Sidebar({ activeTab, setActiveTab, aiConnected }: SidebarProps) {
  const menuItems = [
    { id: "dashboard", label: "Overview", icon: LayoutDashboard, category: "MONITOR" },
    { id: "forecast", label: "Predictive Forecast", icon: Sliders, category: "FORECAST" },
    { id: "eda", label: "Analytics & EDA", icon: LineChart, category: "RESEARCH" },
    { id: "shap", label: "SHAP Explainability", icon: Eye, category: "RESEARCH" },
    { id: "ml", label: "ML Classifiers", icon: Cpu, category: "MODELS" },
    { id: "dl", label: "Deep Learning", icon: Layers, category: "MODELS" },
    { id: "stations", label: "Regional Nodes", icon: Map, category: "SPATIAL" },
    { id: "seasonal", label: "Seasonal Heatmaps", icon: Calendar, category: "SPATIAL" },
    { id: "research", label: "Research Hub", icon: BookOpen, category: "DOCUMENTATION" }
  ];

  const categories = Array.from(new Set(menuItems.map((item) => item.category)));

  return (
    <aside id="sidebar-container" className="w-68 bg-[#0A0E14]/80 backdrop-blur-xl border-r border-white/10 p-5 flex flex-col justify-between h-screen sticky top-0">
      <div className="space-y-6">
        {/* Header Branding */}
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <div className="shrink-0 w-11 h-11 relative">
              <svg className="w-full h-full overflow-visible" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                  <linearGradient id="logo-glow" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#06b6d4" />
                    <stop offset="50%" stopColor="#3b82f6" />
                    <stop offset="100%" stopColor="#6366f1" />
                  </linearGradient>
                  <filter id="logo-shadow" x="-30%" y="-30%" width="160%" height="160%">
                    <feDropShadow dx="0" dy="1" stdDeviation="2.5" floodColor="#06b6d4" floodOpacity="0.5" />
                  </filter>
                </defs>

                {/* Hexagonal container shell */}
                <path d="M20 2 L36 11 V29 L20 38 L4 29 V11 Z" fill="url(#logo-glow)" fillOpacity="0.08" stroke="url(#logo-glow)" strokeWidth="1.5" />
                
                {/* Glowing inner rings */}
                <path d="M20 5 L33 12.5 V27.5 L20 35 L7 27.5 V12.5 Z" stroke="url(#logo-glow)" strokeWidth="0.75" strokeDasharray="3,3" opacity="0.6" />

                {/* Flowing wind vectors (AQI streams) */}
                <path d="M12 16 C15 14, 17 18, 21 16 C25 14, 27 16, 28 16" stroke="#22d3ee" strokeWidth="1.75" strokeLinecap="round" opacity="0.95" />
                <path d="M10 20 C14 18, 17 22, 21 20 C25 18, 27 21, 30 20" stroke="#3b82f6" strokeWidth="1.75" strokeLinecap="round" opacity="0.85" />
                <path d="M13 24 C16 23, 18 25, 21 24 C24 23, 26 24, 27 24" stroke="#6366f1" strokeWidth="1.75" strokeLinecap="round" opacity="0.75" />

                {/* Neural connectivity nodes */}
                <circle cx="20" cy="20" r="3" fill="#ffffff" filter="url(#logo-shadow)" />
                <circle cx="20" cy="20" r="1.2" fill="#05070a" />
              </svg>
            </div>
            <div>
              <h1 className="font-display font-bold text-[10.5px] tracking-wider text-white uppercase leading-tight">
                Scalable AQI Forecasting <span className="text-cyan-400">& XAI System</span>
              </h1>
              <p className="text-[8px] text-white/45 font-mono tracking-tight uppercase">
                Noida Multi-Station AQI (2015–2026)
              </p>
            </div>
          </div>
        </div>

        {/* System Node Badge */}
        <div className="p-3 rounded-xl bg-white/[0.04] border border-white/10 space-y-1.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
              <span className="text-[10px] font-mono text-slate-300 font-semibold">Prediction Engine Online</span>
            </div>
            <span className="text-[9px] font-mono bg-cyan-400/10 text-cyan-400 px-1.5 py-0.5 rounded border border-cyan-400/20 uppercase font-semibold">
              v2.4
            </span>
          </div>
          <div className="flex items-center justify-between text-[9.5px] font-mono text-white/50 pt-1 border-t border-white/5">
            <span>Model: HistGradientBoosting</span>
            <span className="text-emerald-400 font-medium">Loaded</span>
          </div>
          <div className="flex items-center justify-between text-[9.5px] font-mono text-white/50">
            <span>Backend API</span>
            <span className="text-emerald-400 font-medium">Connected</span>
          </div>
          <div className="flex items-center justify-between text-[9.5px] font-mono text-white/40">
            <span>Last Sync</span>
            <span>{new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})}</span>
          </div>
        </div>

        {/* Navigation Groups */}
        <nav className="space-y-4">
          {categories.map((category) => (
            <div key={category} className="space-y-1.5">
              <span className="text-[9px] font-mono font-bold text-white/30 uppercase tracking-widest block px-2">
                {category}
              </span>
              <ul className="space-y-1">
                {menuItems
                  .filter((item) => item.category === category)
                  .map((item) => {
                    const Icon = item.icon;
                    const isActive = activeTab === item.id;
                    return (
                      <li key={item.id}>
                        <button
                          id={`btn-tab-${item.id}`}
                          onClick={() => setActiveTab(item.id as Tab)}
                          className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium font-sans transition-all duration-200 cursor-pointer ${
                            isActive
                              ? "bg-cyan-400/10 text-cyan-400 border border-cyan-400/20 shadow-[0_0_10px_rgba(34,211,238,0.1)]"
                              : "text-slate-400 hover:bg-white/[0.04] hover:text-white border border-transparent"
                          }`}
                        >
                          <Icon className={`w-4 h-4 ${isActive ? "text-cyan-400" : "text-slate-500"}`} />
                          <span className="truncate">{item.label}</span>
                          {item.id === "shap" && (
                            <span className="ml-auto text-[9px] font-mono bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-1 py-0.2 rounded">
                              SHAP
                            </span>
                          )}
                          {item.id === "forecast" && (
                            <span className="ml-auto">
                              <Sparkles className="w-3 h-3 text-cyan-400 animate-pulse" />
                            </span>
                          )}
                        </button>
                      </li>
                    );
                  })}
              </ul>
            </div>
          ))}
        </nav>
      </div>

      {/* Footer Meta */}
      <div className="pt-4 border-t border-white/10">
        <div className="flex items-center justify-between text-[10px] text-white/30 font-mono">
          <span>COPERNICUS L3</span>
          <span className="text-cyan-400/50 font-bold">GRID: 28.58N</span>
        </div>
        <p className="text-[9px] text-white/40 mt-1 leading-relaxed">
          IIT-Delhi & UPPCB Joint Research Project
        </p>
      </div>
    </aside>
  );
}
