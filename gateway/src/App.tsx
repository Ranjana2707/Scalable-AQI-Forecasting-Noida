import React, { useState, useEffect } from "react";
import { Tab, Pollutants, Meteorology } from "./types";
import Sidebar from "./components/Sidebar";
import DashboardView from "./components/DashboardView";
import ForecastView from "./components/ForecastView";
import EdaView from "./components/EdaView";
import ShapView from "./components/ShapView";
import MlView from "./components/MlView";
import DlView from "./components/DlView";
import StationsView from "./components/StationsView";
import SeasonalView from "./components/SeasonalView";
import ResearchView from "./components/ResearchView";

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("dashboard");
  const [selectedStation, setSelectedStation] = useState<string>("sec62");
  
  // Real-time environmental metrics (synchronized across widgets)
  const [pollutants, setPollutants] = useState<Pollutants>({
    pm25: 184,
    pm10: 295,
    no2: 62,
    so2: 8,
    co: 1.8,
    o3: 34,
    nh3: 12
  });

  const [meteorology, setMeteorology] = useState<Meteorology>({
    temperature: 15.4,
    humidity: 82,
    windSpeed: 4.2,
    rainfall: 0
  });

  const [forecastHorizon, setForecastHorizon] = useState<number>(72);
  const [predictedAqi, setPredictedAqi] = useState<number>(184);
  const [aiConnected, setAiConnected] = useState<boolean>(false);
  const [healthStatus, setHealthStatus] = useState<{
    connected: boolean;
    aiEnabled: boolean;
    modelsLoaded: string[];
    lastSync: string;
  }>({
    connected: false,
    aiEnabled: false,
    modelsLoaded: [],
    lastSync: "--:--:--"
  });

  // Check backend server & Gemini connectivity on initial boot
  useEffect(() => {
    async function checkBackend() {
      try {
        const res = await fetch("/api/health");
        const data = await res.json();
        if (data.status === "ok") {
          setHealthStatus({
            connected: true,
            aiEnabled: !!data.aiEnabled,
            modelsLoaded: data.mlModelsLoaded || [],
            lastSync: data.lastSync || "Just now"
          });
          setAiConnected(!!data.aiEnabled);
        }
      } catch (err) {
        console.warn("[SYSTEM] Express server offline.");
        setHealthStatus({
          connected: false,
          aiEnabled: false,
          modelsLoaded: [],
          lastSync: "Offline"
        });
        setAiConnected(false);
      }
    }
    checkBackend();
  }, []);

  // Synchronize predicted AQI for the active station
  useEffect(() => {
    let active = true;
    async function fetchActivePrediction() {
      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pollutants,
            meteorology,
            modelName: "HistGradientBoosting",
            forecastHorizon,
            stationId: selectedStation,
            date: new Date().toISOString().split("T")[0]
          })
        });
        const data = await res.json();
        if (active && data && data.predictedAqi !== undefined) {
          setPredictedAqi(data.predictedAqi);
        }
      } catch (err) {
        console.warn("Failed to sync prediction for selected station", err);
      }
    }
    fetchActivePrediction();
    return () => { active = false; };
  }, [selectedStation, pollutants, meteorology, forecastHorizon]);

  // Map tabs to rendering components
  const renderTabContent = () => {
    switch (activeTab) {
      case "dashboard":
        return (
          <DashboardView
            selectedStation={selectedStation}
            setSelectedStation={setSelectedStation}
            predictedAqi={predictedAqi}
          />
        );
      case "forecast":
        return (
          <ForecastView
            selectedStation={selectedStation}
            setSelectedStation={setSelectedStation}
            predictedAqi={predictedAqi}
            setPredictedAqi={setPredictedAqi}
            pollutants={pollutants}
            setPollutants={setPollutants}
            meteorology={meteorology}
            setMeteorology={setMeteorology}
            forecastHorizon={forecastHorizon}
            setForecastHorizon={setForecastHorizon}
          />
        );
      case "eda":
        return <EdaView />;
      case "shap":
        return (
          <ShapView
            predictedAqi={predictedAqi}
            pollutants={pollutants}
            meteorology={meteorology}
          />
        );
      case "ml":
        return <MlView />;
      case "dl":
        return <DlView />;
      case "stations":
        return <StationsView />;
      case "seasonal":
        return <SeasonalView />;
      case "research":
        return (
          <ResearchView
            pollutants={pollutants}
            meteorology={meteorology}
            forecastHorizon={forecastHorizon}
          />
        );
      default:
        return (
          <DashboardView
            selectedStation={selectedStation}
            setSelectedStation={setSelectedStation}
            predictedAqi={predictedAqi}
          />
        );
    }
  };

  return (
    <div className="flex bg-[#05070A] min-h-screen text-slate-100 select-none antialiased">
      {/* Sidebar navigation */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        aiConnected={aiConnected}
        healthStatus={healthStatus}
      />

      {/* Main Viewport */}
      <main className="flex-1 p-6 lg:p-8 overflow-y-auto max-h-screen">
        <div className="max-w-7xl mx-auto space-y-6">
          {renderTabContent()}
        </div>
      </main>
    </div>
  );
}
