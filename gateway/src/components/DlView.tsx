import React, { useState } from "react";
import { DL_METRICS } from "../data";
import { Cpu, Layers, Activity, Server, RefreshCw, Zap, Award } from "lucide-react";

export default function DlView() {
  const [selectedModel, setSelectedModel] = useState("CNN-LSTM Hybrid");
  const [simulationSpeed, setSimulationSpeed] = useState("Fast (1.0s)");
  
  // LSTM Node Activation simulation (6x6 matrix of weights)
  const [activations, setActivations] = useState<number[][]>([
    [0.85, 0.12, 0.45, 0.92, 0.05, 0.62],
    [0.15, 0.72, 0.08, 0.54, 0.88, 0.22],
    [0.65, 0.28, 0.94, 0.11, 0.42, 0.78],
    [0.34, 0.82, 0.51, 0.76, 0.15, 0.48],
    [0.91, 0.04, 0.68, 0.23, 0.81, 0.55],
    [0.22, 0.58, 0.19, 0.89, 0.32, 0.71]
  ]);

  const handleShuffleActivations = () => {
    // Re-simulate neural activations
    const shuffled = activations.map((row) =>
      row.map(() => Number((Math.random() * 0.95 + 0.05).toFixed(2)))
    );
    setActivations(shuffled);
  };

  const activeModelStats = DL_METRICS.find(m => m.name === selectedModel) || DL_METRICS[0];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-widest font-semibold">Temporal neural networks</span>
          <h2 className="text-xl font-display font-bold text-white mt-1">Deep Learning Recurrent Networks</h2>
          <p className="text-xs text-slate-400">Examine temporal sequence models, cell gate weights, and real-time GPU tensor speed performance ratios.</p>
        </div>

        <button
          onClick={handleShuffleActivations}
          className="px-4 py-2 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 text-white font-mono text-xs font-bold uppercase shadow-[0_0_15px_rgba(99,102,241,0.4)] cursor-pointer transition duration-200"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Stimulate Node Activations
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Model Comparative Metrics Block */}
        <div className="col-span-1 lg:col-span-8 bg-white/[0.03] border border-white/10 rounded-2xl p-5 flex flex-col justify-between backdrop-blur-md">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-mono uppercase tracking-widest text-white/80 font-semibold flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-cyan-400" />
                Deep Learning Model Benchmarks
              </h3>
              <span className="text-[10px] text-white/40 font-mono">Sequence inputs (24h lookback)</span>
            </div>

            {/* DL Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono text-left">
                <thead>
                  <tr className="border-b border-white/10 text-white/40 pb-2">
                    <th className="py-2.5 font-bold uppercase text-[10px]">Model name</th>
                    <th className="py-2.5 font-bold uppercase text-[10px] text-center">MAE ↓</th>
                    <th className="py-2.5 font-bold uppercase text-[10px] text-center">RMSE ↓</th>
                    <th className="py-2.5 font-bold uppercase text-[10px] text-center">R² Score ↑</th>
                    <th className="py-2.5 font-bold uppercase text-[10px] text-center">Epochs</th>
                    <th className="py-2.5 font-bold uppercase text-[10px] text-right">Inference</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {DL_METRICS.map((m) => (
                    <tr
                      key={m.name}
                      onClick={() => setSelectedModel(m.name)}
                      className={`hover:bg-white/[0.04] cursor-pointer transition duration-150 ${
                        selectedModel === m.name ? "bg-indigo-500/10" : ""
                      }`}
                    >
                      <td className="py-3 font-semibold text-slate-200 flex items-center gap-2">
                        {m.name}
                        {m.name === "CNN-LSTM Hybrid" && (
                          <span className="text-[8px] font-bold font-sans bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 px-1.5 py-0.5 rounded flex items-center gap-1">
                            <Award className="w-2.5 h-2.5" /> ACCURACY KING
                          </span>
                        )}
                      </td>
                      <td className="py-3 text-center text-slate-300">{m.mae.toFixed(2)}</td>
                      <td className="py-3 text-center text-slate-300">{m.rmse.toFixed(2)}</td>
                      <td className="py-3 text-center text-indigo-400 font-bold">{m.r2.toFixed(3)}</td>
                      <td className="py-3 text-center text-slate-400">150 Epochs</td>
                      <td className="py-3 text-right text-slate-400">{m.inferenceTime.toFixed(2)}ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Detailed Node Information */}
          <div className="mt-6 p-4 rounded-xl bg-[#0A0E14] border border-white/10">
            <h4 className="text-[11px] font-mono font-bold uppercase text-slate-300 mb-1">Architecture Narrative: {activeModelStats.name}</h4>
            <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
              {activeModelStats.name === "CNN-LSTM Hybrid" 
                ? "Leverages 1D convolutional layers to extract local spatial features from pollutant timelines, feeding a recurrent LSTM structure to map long-term temporal trends." 
                : "Standard recurrent neural gating model. Demonstrates reliable sequence learning on regional air quality indicators, showing marginal accuracy decay relative to the hybrid network."}
            </p>
          </div>
        </div>

        {/* Right Side: LSTM Gate Activations Matrix & Hardware Monitor */}
        <div className="col-span-1 lg:col-span-4 flex flex-col gap-6">
          
          {/* LSTM Activations Matrix SVG */}
          <div className="p-5 rounded-2xl bg-[#0A0E14] border border-white/10 flex flex-col justify-between backdrop-blur-md">
            <div>
              <div className="flex items-center justify-between mb-3.5">
                <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-white/80 flex items-center gap-1.5">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  Neural Cell Gate Weights
                </h4>
                <span className="text-[9px] font-mono text-cyan-400">LSTM Gates</span>
              </div>

              {/* Activation Grid */}
              <div className="grid grid-cols-6 gap-2 bg-[#05070A] p-4 rounded-xl border border-white/10">
                {activations.map((row, rIdx) =>
                  row.map((val, cIdx) => {
                    // Opacity reflects activation value
                    const glowStyle = {
                      backgroundColor: `rgba(99, 102, 241, ${val})`,
                      boxShadow: val > 0.7 ? `0 0 10px rgba(99, 102, 241, ${val * 0.5})` : "none"
                    };

                    return (
                      <div
                        key={`${rIdx}-${cIdx}`}
                        style={glowStyle}
                        title={`Node weight: ${val}`}
                        className={`aspect-square rounded flex items-center justify-center text-[8px] font-mono font-semibold transition-all duration-300 ${
                          val > 0.65 ? "text-slate-100 font-bold" : "text-transparent"
                        } border border-indigo-500/10 cursor-pointer hover:scale-110`}
                      >
                        {Math.round(val * 10)}
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            <p className="text-[10px] text-white/30 font-mono text-center mt-3 pt-3 border-t border-white/10 leading-normal">
              Activations represent real-time recurrent weight filters.
            </p>
          </div>

          {/* Hardware Acceleration Panel */}
          <div className="p-5 rounded-2xl bg-white/[0.03] border border-white/10 space-y-4 backdrop-blur-md">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Server className="w-4 h-4 text-indigo-400" />
                VRAM / Tensor Performance
              </h4>
              <span className="text-[9px] font-mono text-cyan-400 bg-cyan-500/10 px-1.5 py-0.2 rounded border border-cyan-500/20 uppercase">RTX-4090 ACTIVE</span>
            </div>

            <div className="space-y-2.5 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-slate-400">CUDA Threads Allocated</span>
                <span className="text-slate-200">16,384 Threads</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Total VRAM Allocation</span>
                <span className="text-slate-200">1.82 GB / 24 GB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">GPU Core Temp</span>
                <span className="text-amber-400">54.2 °C</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Inference Core Speed</span>
                <span className="text-indigo-400 font-semibold">142.5 TFLOPS</span>
              </div>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
