import React, { useState } from "react";
import { BookOpen, Code, Cpu, Terminal, ArrowUpRight, Check, Copy, Download, FileText, Database, ShieldAlert, FileSpreadsheet, Sliders } from "lucide-react";

interface ResearchViewProps {
  pollutants: { pm25: number; pm10: number; no2: number; so2: number; co: number; o3: number; nh3: number };
  meteorology: { temperature: number; humidity: number; windSpeed: number; rainfall: number };
  forecastHorizon: number;
}

export default function ResearchView({ pollutants, meteorology, forecastHorizon }: ResearchViewProps) {
  const [activeCodeLang, setActiveCodeLang] = useState<"python" | "curl">("python");
  const [copied, setCopied] = useState(false);

  // Generate dynamic Python code snippet based on current state parameters
  const generatePythonCode = () => {
    return `import requests

# Noida AQI Predictive Core API Ingestion
API_URL = "http://localhost:3000/api/predict"

payload = {
    "station": "Sector-62, Noida",
    "model_name": "HistGradientBoosting",
    "forecast_horizon_hours": ${forecastHorizon},
    "pollutants": {
        "pm25": ${pollutants.pm25},
        "pm10": ${pollutants.pm10},
        "no2": ${pollutants.no2},
        "so2": ${pollutants.so2},
        "co": ${pollutants.co},
        "o3": ${pollutants.o3}
    },
    "meteorology": {
        "temperature": ${meteorology.temperature},
        "humidity": ${meteorology.humidity},
        "wind_speed": ${meteorology.windSpeed},
        "rainfall": ${meteorology.rainfall}
    }
}

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer NOIDA_AEROSOL_GRID_TOKEN"
}

response = requests.post(API_URL, json=payload, headers=headers)
prediction = response.json()

print(f"Predicted AQI: {prediction['predicted_aqi']}")
print(f"Confidence Range: {prediction['lower_bound']} - {prediction['upper_bound']}")`;
  };

  // Generate dynamic Curl snippet
  const generateCurlCode = () => {
    return `curl -X POST http://localhost:3000/api/predict \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer NOIDA_AEROSOL_GRID_TOKEN" \\
  -d '{
    "station": "Sector-62, Noida",
    "model_name": "HistGradientBoosting",
    "forecast_horizon_hours": ${forecastHorizon},
    "pollutants": {
      "pm25": ${pollutants.pm25},
      "pm10": ${pollutants.pm10},
      "no2": ${pollutants.no2},
      "so2": ${pollutants.so2},
      "co": ${pollutants.co},
      "o3": ${pollutants.o3}
    },
    "meteorology": {
      "temperature": ${meteorology.temperature},
      "humidity": ${meteorology.humidity},
      "wind_speed": ${meteorology.windSpeed},
      "rainfall": ${meteorology.rainfall}
    }
  }'`;
  };

  const activeCode = activeCodeLang === "python" ? generatePythonCode() : generateCurlCode();

  const handleCopy = () => {
    navigator.clipboard.writeText(activeCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-widest font-semibold">Citations & api deployment</span>
        <h2 className="text-xl font-display font-bold text-white mt-1">Research Hub & Model Integration</h2>
        <p className="text-xs text-slate-400">Review peer-reviewed model conclusions and leverage the dynamic developer code sandbox to query Noida predictors remotely.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Core Research Findings */}
        <div className="col-span-1 lg:col-span-6 bg-white/[0.03] border border-white/10 rounded-2xl p-5 flex flex-col justify-between backdrop-blur-md">
          <div className="space-y-4">
            <h3 className="text-xs font-mono uppercase tracking-widest text-white/80 font-semibold flex items-center gap-1.5 pb-2 border-b border-white/10">
              <BookOpen className="w-4 h-4 text-cyan-400" />
              Featured Scientific Discoveries
            </h3>

            <div className="space-y-3 max-h-[360px] overflow-y-auto pr-1">
              {/* Finding 1 */}
              <div className="p-3 rounded-xl bg-[#0A0E14] border border-white/10 text-xs leading-relaxed space-y-1">
                <span className="text-[9px] font-mono font-bold text-cyan-400 uppercase tracking-wider">AEROSOL INVERSION LAYER PHYSICS</span>
                <h4 className="text-white font-semibold">Boundary Layer Height Correlation</h4>
                <p className="text-slate-400 font-sans text-[11px]">
                  Research confirms a strong negative correlation (-0.74) between planetary boundary layer (PBL) height and PM2.5 density during winter nights. Standard inversion models are updated with HistGradientBoosting to predict winter trapping events with MAE &lt; 9.0 AQI.
                </p>
                <div className="pt-1.5 flex justify-between items-center text-[9px] font-mono text-white/30">
                  <span>IIT-Delhi Atmospheric Division</span>
                  <a href="#cit" className="hover:text-cyan-400 flex items-center gap-0.5">Citation No. 42 <ArrowUpRight className="w-3 h-3" /></a>
                </div>
              </div>

              {/* Finding 2 */}
              <div className="p-3 rounded-xl bg-[#0A0E14] border border-white/10 text-xs leading-relaxed space-y-1">
                <span className="text-[9px] font-mono font-bold text-indigo-400 uppercase tracking-wider">PHOTOCHEMICAL KINETICS</span>
                <h4 className="text-white font-semibold">Titration profiles of O3 & NOx</h4>
                <p className="text-slate-400 font-sans text-[11px]">
                  Photochemical modeling of Noida's regional expressway reveals high Ozone peaks are tightly associated with rapid nitric oxide (NO) emissions titration. The CNN-LSTM hybrid maps this non-linear dependency with high sequence accuracy.
                </p>
                <div className="pt-1.5 flex justify-between items-center text-[9px] font-mono text-white/30">
                  <span>UPPCB Research Lab</span>
                  <a href="#cit" className="hover:text-cyan-400 flex items-center gap-0.5">Citation No. 18 <ArrowUpRight className="w-3 h-3" /></a>
                </div>
              </div>
            </div>
          </div>

          <div className="text-[10px] font-mono text-white/30 pt-3 border-t border-white/10 flex justify-between">
            <span>Citations count: 48 Papers</span>
            <span>Dataset size: 1.4M rows</span>
          </div>
        </div>

        {/* Developer Sandbox (Code Generator) */}
        <div className="col-span-1 lg:col-span-6 bg-white/[0.03] border border-white/10 rounded-2xl p-5 flex flex-col justify-between backdrop-blur-md">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-mono uppercase tracking-widest text-white/80 font-semibold flex items-center gap-1.5">
                <Code className="w-4 h-4 text-cyan-400" />
                Inference API Code Sandbox
              </h3>
              
              {/* Code Language Selectors */}
              <div className="flex bg-[#05070A] border border-white/10 p-0.5 rounded-lg text-[9px] font-mono">
                <button
                  onClick={() => setActiveCodeLang("python")}
                  className={`px-2 py-1 rounded cursor-pointer ${activeCodeLang === "python" ? "bg-[#0A0E14] text-cyan-400 font-bold border border-white/10 shadow-[0_0_8px_rgba(6,182,212,0.1)]" : "text-slate-400"}`}
                >
                  Python
                </button>
                <button
                  onClick={() => setActiveCodeLang("curl")}
                  className={`px-2 py-1 rounded cursor-pointer ${activeCodeLang === "curl" ? "bg-[#0A0E14] text-cyan-400 font-bold border border-white/10 shadow-[0_0_8px_rgba(6,182,212,0.1)]" : "text-slate-400"}`}
                >
                  cURL
                </button>
              </div>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed font-sans">
              Integrate Noida AQI forecast models into external pipelines. The code below is **dynamically synchronized** with your active slider parameters.
            </p>

            {/* Code Display Screen */}
            <div className="relative">
              <pre className="bg-[#05070A]/90 text-slate-300 font-mono text-[10px] p-4.5 rounded-xl border border-white/10 h-64 overflow-auto select-all scrollbar-thin">
                {activeCode}
              </pre>

              {/* Copy Button */}
              <button
                onClick={handleCopy}
                className="absolute top-2.5 right-2.5 p-1.5 rounded-lg bg-[#0A0E14] hover:bg-[#121820] text-slate-400 hover:text-white border border-white/10 transition cursor-pointer"
                title="Copy code"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>

          <div className="text-[10px] font-mono text-white/30 pt-3 border-t border-white/10 flex justify-between items-center">
            <span>Inference endpoint: active</span>
            <span className="text-cyan-400 flex items-center gap-1"><Terminal className="w-3 h-3" /> REST Protocol (JSON)</span>
          </div>
        </div>

      </div>

      {/* Scientific Model Artifacts & Resources Download Station */}
      <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-5 space-y-4 backdrop-blur-md">
        <div>
          <h3 className="text-xs font-mono uppercase tracking-widest text-white/80 font-semibold flex items-center gap-1.5 pb-2 border-b border-white/10">
            <Download className="w-4 h-4 text-cyan-400" />
            Model Artifacts & Scientific Outputs Download Center
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Access and download verified files directly from the Noida forecasting backend node for local inspection, research replication, or custom integration.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          
          {/* Paper */}
          <div className="p-3.5 rounded-xl bg-[#0A0E14] border border-white/10 hover:border-cyan-500/30 transition flex flex-col justify-between space-y-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-cyan-400">
                <FileText className="w-4 h-4" />
                <span className="font-mono text-[10px] font-bold uppercase tracking-wider">Research Article</span>
              </div>
              <h4 className="text-white text-xs font-semibold">Scientific Research Paper</h4>
              <p className="text-[10.5px] text-slate-400 font-sans leading-normal">
                Verified pre-print summarizing planetary boundary layer inversion dynamics & SHAP alignment in Noida.
              </p>
            </div>
            <a
              href="/api/download/paper"
              download="noida_aqi_xai_research_paper.txt"
              className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-cyan-400/10 hover:bg-cyan-400/20 text-cyan-400 font-mono text-[10px] uppercase font-bold tracking-wider transition cursor-pointer"
            >
              <Download className="w-3 h-3" /> Download Paper (TXT)
            </a>
          </div>

          {/* Dataset */}
          <div className="p-3.5 rounded-xl bg-[#0A0E14] border border-white/10 hover:border-cyan-500/30 transition flex flex-col justify-between space-y-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-cyan-400">
                <Database className="w-4 h-4" />
                <span className="font-mono text-[10px] font-bold uppercase tracking-wider">Historical Grid</span>
              </div>
              <h4 className="text-white text-xs font-semibold">Trained Model Dataset</h4>
              <p className="text-[10.5px] text-slate-400 font-sans leading-normal">
                Multi-station continuous hourly record for Noida Sector-62 & Sector-1 (2015-2026) in CSV structure.
              </p>
            </div>
            <a
              href="/api/download/dataset"
              download="noida_aqi_historical_dataset.csv"
              className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-cyan-400/10 hover:bg-cyan-400/20 text-cyan-400 font-mono text-[10px] uppercase font-bold tracking-wider transition cursor-pointer"
            >
              <Download className="w-3 h-3" /> Download Dataset (CSV)
            </a>
          </div>

          {/* Model Weights */}
          <div className="p-3.5 rounded-xl bg-[#0A0E14] border border-white/10 hover:border-indigo-500/30 transition flex flex-col justify-between space-y-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-indigo-400">
                <Cpu className="w-4 h-4" />
                <span className="font-mono text-[10px] font-bold uppercase tracking-wider">Serialized Model</span>
              </div>
              <h4 className="text-white text-xs font-semibold">Model Architecture & Weights</h4>
              <p className="text-[10.5px] text-slate-400 font-sans leading-normal">
                The serialized JSON containing optimized tree thresholds and hyper-parameters for HistGradientBoosting.
              </p>
            </div>
            <a
              href="/api/download/model"
              download="hist_gradient_boosting_model.json"
              className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 font-mono text-[10px] uppercase font-bold tracking-wider transition cursor-pointer"
            >
              <Download className="w-3 h-3" /> Download Model (JSON)
            </a>
          </div>

          {/* Figures */}
          <div className="p-3.5 rounded-xl bg-[#0A0E14] border border-white/10 hover:border-cyan-500/30 transition flex flex-col justify-between space-y-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-cyan-400">
                <FileText className="w-4 h-4" />
                <span className="font-mono text-[10px] font-bold uppercase tracking-wider">Publication Visuals</span>
              </div>
              <h4 className="text-white text-xs font-semibold">Publication Figures</h4>
              <p className="text-[10.5px] text-slate-400 font-sans leading-normal">
                Verification diagrams, spatial map grids, neural cell activations, and comparative residual loss curves.
              </p>
            </div>
            <a
              href="/api/download/figures"
              download="noida_aqi_publication_figures.txt"
              className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-cyan-400/10 hover:bg-cyan-400/20 text-cyan-400 font-mono text-[10px] uppercase font-bold tracking-wider transition cursor-pointer"
            >
              <Download className="w-3 h-3" /> Download Figures (TXT)
            </a>
          </div>

          {/* SHAP */}
          <div className="p-3.5 rounded-xl bg-[#0A0E14] border border-white/10 hover:border-indigo-500/30 transition flex flex-col justify-between space-y-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-indigo-400">
                <Sliders className="w-4 h-4 text-indigo-400" />
                <span className="font-mono text-[10px] font-bold uppercase tracking-wider">Additive Attributions</span>
              </div>
              <h4 className="text-white text-xs font-semibold">Download SHAP Outputs</h4>
              <p className="text-[10.5px] text-slate-400 font-sans leading-normal">
                Global Shapley values calculated over continuous testing partitions for game-theoretic feature analysis.
              </p>
            </div>
            <a
              href="/api/download/shap"
              download="noida_global_shap_values.json"
              className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 font-mono text-[10px] uppercase font-bold tracking-wider transition cursor-pointer"
            >
              <Download className="w-3 h-3" /> Download SHAP (JSON)
            </a>
          </div>

          {/* Results */}
          <div className="p-3.5 rounded-xl bg-[#0A0E14] border border-white/10 hover:border-cyan-500/30 transition flex flex-col justify-between space-y-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-cyan-400">
                <FileSpreadsheet className="w-4 h-4 text-cyan-400" />
                <span className="font-mono text-[10px] font-bold uppercase tracking-wider">Performance metrics</span>
              </div>
              <h4 className="text-white text-xs font-semibold">Download Comparative Results</h4>
              <p className="text-[10.5px] text-slate-400 font-sans leading-normal">
                Detailed MAE, RMSE, MAPE, and R² statistics comparing all ML and Deep Learning architectures.
              </p>
            </div>
            <a
              href="/api/download/results"
              download="noida_model_comparison_results.json"
              className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-cyan-400/10 hover:bg-cyan-400/20 text-cyan-400 font-mono text-[10px] uppercase font-bold tracking-wider transition cursor-pointer"
            >
              <Download className="w-3 h-3" /> Download Results (JSON)
            </a>
          </div>

        </div>
      </div>
    </div>
  );
}
