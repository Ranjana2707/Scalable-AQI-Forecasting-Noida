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

  // Global Time & Date State Selection
  const [selectedDate, setSelectedDate] = useState(() => {
    const d = new Date();
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  });
  const [selectedTime, setSelectedTime] = useState(() => {
    const d = new Date();
    const hours = String(d.getHours()).padStart(2, "0");
    const minutes = String(d.getMinutes()).padStart(2, "0");
    return `${hours}:${minutes}`;
  });
  const [isLiveTime, setIsLiveTime] = useState(true);
  const [activeModel, setActiveModel] = useState("HistGradientBoosting");
  const [sensorDrift, setSensorDrift] = useState(false);

  const [historical, setHistorical] = useState<number[]>([142, 148, 156, 150, 162, 175, 184]);
  const [forecast, setForecast] = useState<number[]>([190, 195, 204, 210, 220, 230]);
  const [backendLoading, setBackendLoading] = useState(false);
  const [backendError, setBackendError] = useState(false);

  // Check backend server & Gemini connectivity on initial boot
  useEffect(() => {
    async function checkBackend() {
      try {
        const res = await fetch("/api/health");
        const data = await res.json();
        if (data.status === "ok") {
          setAiConnected(data.aiEnabled);
        }
      } catch (err) {
        console.warn("[SYSTEM] Express server not fully booted or offline. Using high-fidelity client simulation fallback.");
        setAiConnected(false);
      }
    }
    checkBackend();
  }, []);

  // Auto-update time/date every 10 seconds to match system clock, unless manually edited
  useEffect(() => {
    if (!isLiveTime) return;

    const updateTime = () => {
      const d = new Date();
      const year = d.getFullYear();
      const month = String(d.getMonth() + 1).padStart(2, "0");
      const day = String(d.getDate()).padStart(2, "0");
      const dateStr = `${year}-${month}-${day}`;

      const hours = String(d.getHours()).padStart(2, "0");
      const minutes = String(d.getMinutes()).padStart(2, "0");
      const timeStr = `${hours}:${minutes}`;

      setSelectedDate(dateStr);
      setSelectedTime(timeStr);
    };

    updateTime();
    const interval = setInterval(updateTime, 10000);
    return () => clearInterval(interval);
  }, [isLiveTime]);

  // Unified prediction hook running in the background
  useEffect(() => {
    let active = true;
    const controller = new AbortController();

    async function fetchForecast() {
      setBackendLoading(true);
      setBackendError(false);
      try {
        const bodyPollutants = { ...pollutants };
        if (sensorDrift) {
          bodyPollutants.pm25 += 25;
        }

        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pollutants: bodyPollutants,
            meteorology,
            modelName: activeModel,
            forecastHorizon,
            stationId: selectedStation,
            date: selectedDate,
            time: selectedTime
          }),
          signal: controller.signal
        });
        const data = await res.json();
        if (active && data) {
          if (data.predictedAqi !== undefined) {
            setPredictedAqi(data.predictedAqi);
            setHistorical(data.historical);
            setForecast(data.forecast);
          } else {
            setBackendError(true);
          }
        }
      } catch (err: any) {
        if (err.name !== "AbortError") {
          console.error("Prediction API fetch failed:", err);
          if (active) setBackendError(true);
        }
      } finally {
        if (active) setBackendLoading(false);
      }
    }

    fetchForecast();

    return () => {
      active = false;
      controller.abort();
    };
  }, [pollutants, meteorology, activeModel, sensorDrift, forecastHorizon, selectedStation, selectedDate, selectedTime]);

  // Map tabs to rendering components
  const renderTabContent = () => {
    switch (activeTab) {
      case "dashboard":
        return (
          <DashboardView
            selectedStation={selectedStation}
            setSelectedStation={setSelectedStation}
            predictedAqi={predictedAqi}
            forecastHorizon={forecastHorizon}
            setForecastHorizon={setForecastHorizon}
            selectedDate={selectedDate}
            selectedTime={selectedTime}
            activeModel={activeModel}
            historical={historical}
            forecast={forecast}
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
            selectedDate={selectedDate}
            setSelectedDate={setSelectedDate}
            selectedTime={selectedTime}
            setSelectedTime={setSelectedTime}
            activeModel={activeModel}
            setActiveModel={setActiveModel}
            sensorDrift={sensorDrift}
            setSensorDrift={setSensorDrift}
            backendLoading={backendLoading}
            backendError={backendError}
            historical={historical}
            setHistorical={setHistorical}
            forecast={forecast}
            setForecast={setForecast}
            isLiveTime={isLiveTime}
            setIsLiveTime={setIsLiveTime}
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
            activeModel={activeModel}
            selectedStation={selectedStation}
            selectedDate={selectedDate}
            selectedTime={selectedTime}
            forecastHorizon={forecastHorizon}
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
            forecastHorizon={forecastHorizon}
            setForecastHorizon={setForecastHorizon}
            selectedDate={selectedDate}
            selectedTime={selectedTime}
            activeModel={activeModel}
            historical={historical}
            forecast={forecast}
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
