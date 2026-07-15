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
  healthStatus?: {
    connected: boolean;
    aiEnabled: boolean;
    modelsLoaded: string[];
    lastSync: string;
  };
}

export default function Sidebar({ activeTab, setActiveTab, aiConnected, healthStatus }: SidebarProps) {
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
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold shadow-[0_0_15px_rgba(6,182,212,0.4)]">
              AQI
            </div>
            <div>
              <h1 className="font-display font-bold text-[11px] tracking-wider text-white uppercase leading-tight">
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
            <span className={healthStatus?.modelsLoaded.includes("HistGradientBoosting") ? "text-emerald-400 font-medium" : "text-rose-500 font-medium"}>
              {healthStatus?.modelsLoaded.includes("HistGradientBoosting") ? "Loaded" : "Offline"}
            </span>
          </div>
          <div className="flex items-center justify-between text-[9.5px] font-mono text-white/50">
            <span>Backend API</span>
            <span className={healthStatus?.connected ? "text-emerald-400 font-medium" : "text-rose-500 font-medium"}>
              {healthStatus?.connected ? "Connected" : "Offline"}
            </span>
          </div>
          <div className="flex items-center justify-between text-[9.5px] font-mono text-white/40">
            <span>Last Sync</span>
            <span>{healthStatus?.lastSync || "--:--:--"}</span>
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
