import React, { useState, useEffect, useRef } from "react";
import { ML_METRICS } from "../data";
import { Cpu, Terminal, RefreshCw, Check, Play, ShieldAlert, Award } from "lucide-react";

export default function MlView() {
  const [selectedModel, setSelectedModel] = useState("HistGradientBoosting");
  const [training, setTraining] = useState(false);
  const [terminalLogs, setTerminalLogs] = useState<string[]>([
    "[CORE] Atmospheric forecasting ML engine initialized.",
    "[CORE] Production model: HistGradientBoosting (R² = 0.941, MAE = 8.84)",
    "Ready for training commands."
  ]);

  const terminalEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll terminal logs to bottom
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [terminalLogs]);

  // Model Retrain Simulator
  const handleRetrain = () => {
    setTraining(true);
    setTerminalLogs(prev => [
      ...prev,
      `[COMMAND] Initializing training run for model: ${selectedModel}`,
      `[PREPROC] Splitting Noida continuous grid datasets into Train/Val/Test (80/10/10)`,
      `[PREPROC] Engineering 24h lag series & meteorological dispersion features...`,
      `[TRAIN] Loading HistGradientBoosting solver...`
    ]);

    let step = 0;
    const steps = [
      `[EPOCH 1] Training loss: 42.15 | Validation loss: 44.82`,
      `[EPOCH 2] Training loss: 24.12 | Validation loss: 26.54`,
      `[EPOCH 3] Training loss: 14.58 | Validation loss: 16.12`,
      `[EPOCH 4] Training loss: 9.85  | Validation loss: 11.24`,
      `[EPOCH 5] Training loss: 8.12  | Validation loss: 9.15  | Learning rate decayed to 0.05`,
      `[EPOCH 6] Training loss: 7.92  | Validation loss: 8.88  | Early stopping triggered (Patience=3)`,
      `[EVAL] Test set results for ${selectedModel}:`,
      `       - MAE: ${(selectedModel === "HistGradientBoosting" ? 8.84 : 9.60)} AQI`,
      `       - RMSE: ${(selectedModel === "HistGradientBoosting" ? 12.15 : 13.10)} AQI`,
      `       - R-squared (R²): ${(selectedModel === "HistGradientBoosting" ? 0.941 : 0.929)}`,
      `[SUCCESS] Model weight files serialized to /weights/${selectedModel}_weights.bin`,
      `[SYSTEM] Production route updated to point to newly compiled weights successfully!`
    ];

    const timer = setInterval(() => {
      if (step < steps.length) {
        setTerminalLogs(prev => [...prev, steps[step]]);
        step++;
      } else {
        clearInterval(timer);
        setTraining(false);
      }
    }, 850);
  };

  const activeModelStats = ML_METRICS.find(m => m.name === selectedModel) || ML_METRICS[0];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-widest font-semibold">Classification algorithms</span>
          <h2 className="text-xl font-display font-bold text-white mt-1">Machine Learning Regression Classifiers</h2>
          <p className="text-xs text-slate-400">Review training latency, mathematical accuracy metrics, and compile optimized model weights on the fly.</p>
        </div>

        <button
          onClick={handleRetrain}
          disabled={training}
          className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 font-mono text-xs font-bold flex items-center gap-2 shadow-[0_0_15px_rgba(6,182,212,0.4)] cursor-pointer transition duration-200"
        >
          {training ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              Retraining model...
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5" />
              Retrain selected Model
            </>
          )}
        </button>
      </div>

      {/* Model Selection and Comparative Metrics Table */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Metric Comparison Table */}
        <div className="col-span-1 lg:col-span-8 bg-white/[0.03] border border-white/10 rounded-2xl p-5 flex flex-col justify-between backdrop-blur-md">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs font-mono uppercase tracking-widest text-white/80 font-semibold flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-cyan-400" />
                Regressor Model Benchmark Matrix
              </h3>
              <span className="text-[10px] text-white/40 font-mono">Test split evaluation</span>
            </div>

            {/* Benchmark Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono text-left">
                <thead>
                  <tr className="border-b border-white/10 text-white/40 pb-2">
                    <th className="py-2.5 font-bold uppercase text-[10px]">Model name</th>
                    <th className="py-2.5 font-bold uppercase text-[10px] text-center">MAE ↓</th>
                    <th className="py-2.5 font-bold uppercase text-[10px] text-center">RMSE ↓</th>
                    <th className="py-2.5 font-bold uppercase text-[10px] text-center">R² Score ↑</th>
                    <th className="py-2.5 font-bold uppercase text-[10px] text-center">Train Time</th>
                    <th className="py-2.5 font-bold uppercase text-[10px] text-right">Inference</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {ML_METRICS.map((m) => (
                    <tr
                      key={m.name}
                      onClick={() => setSelectedModel(m.name)}
                      className={`hover:bg-white/[0.04] cursor-pointer transition duration-150 ${
                        selectedModel === m.name ? "bg-cyan-400/10" : ""
                      }`}
                    >
                      <td className="py-3 font-semibold text-slate-200 flex items-center gap-2">
                        {m.name}
                        {m.isProduction && (
                          <span className="text-[8px] font-bold font-sans bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-1.5 py-0.5 rounded flex items-center gap-1">
                            <Award className="w-2.5 h-2.5" /> PRODUCTION
                          </span>
                        )}
                      </td>
                      <td className="py-3 text-center text-slate-300">{m.mae.toFixed(2)}</td>
                      <td className="py-3 text-center text-slate-300">{m.rmse.toFixed(2)}</td>
                      <td className="py-3 text-center text-cyan-400 font-bold">{m.r2.toFixed(3)}</td>
                      <td className="py-3 text-center text-slate-400">{m.trainTime.toFixed(1)}s</td>
                      <td className="py-3 text-right text-slate-400">{m.inferenceTime.toFixed(2)}ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Model comparison visual helper */}
          <div className="mt-6 p-4 rounded-xl bg-[#0A0E14] border border-white/10 flex items-center justify-between text-xs">
            <div className="space-y-1">
              <span className="text-slate-400 block font-semibold">Active Selection: {activeModelStats.name}</span>
              <p className="text-slate-500 text-[11px] font-sans">
                {activeModelStats.isProduction 
                  ? "HistGradientBoosting achieves the optimal speed/accuracy trade-off on tabular air quality data." 
                  : `Model selected for diagnostic simulation. R² score shows ${((1 - activeModelStats.r2) * 100).toFixed(1)}% variance error from peak production model.`}
              </p>
            </div>
          </div>
        </div>

        {/* Right Side: Training Log Terminal (Developer sandbox) */}
        <div className="col-span-1 lg:col-span-4 flex flex-col justify-between gap-6">
          
          <div className="p-5 rounded-2xl bg-white/[0.03] border border-white/10 flex flex-col justify-between h-full relative overflow-hidden backdrop-blur-md">
            <div className="absolute top-2 right-2 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-red-500" />
              <span className="w-2 h-2 rounded-full bg-yellow-500" />
              <span className="w-2 h-2 rounded-full bg-green-500" />
            </div>

            <div className="space-y-3 flex-1 flex flex-col">
              <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-white/50 flex items-center gap-1.5 pb-2 border-b border-white/10">
                <Terminal className="w-4 h-4 text-cyan-400" />
                Training shell output
              </h4>

              {/* Log stream viewport */}
              <div id="terminal-screen" className="flex-1 bg-[#05070A]/90 p-3 rounded-lg border border-white/10 font-mono text-[10px] leading-relaxed text-slate-400 overflow-y-auto space-y-1.5 h-[230px] select-none">
                {terminalLogs.map((log, idx) => (
                  <div key={idx} className={log.startsWith("[EPOCH") ? "text-cyan-400" : log.startsWith("[SUCCESS") ? "text-emerald-400 font-semibold" : log.startsWith("[COMMAND") ? "text-white" : ""}>
                    {log}
                  </div>
                ))}
                {training && (
                  <div className="flex items-center gap-1 text-cyan-400 animate-pulse font-semibold">
                    <span>█</span> Training...
                  </div>
                )}
                <div ref={terminalEndRef} />
              </div>
            </div>

            <div className="text-[10px] font-mono text-white/30 text-center mt-3 pt-3 border-t border-white/10 flex justify-between">
              <span>Device: CPU/AVX2</span>
              <span>Memory limit: 2048MB</span>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
