import { ModelMetric, SHAPFeature, DataPoint } from "./types";

export const STATIONS = [
  { id: "sec62", name: "Sector-62, Noida", lat: 28.6241, lng: 77.3732, areaType: "Industrial/Residential" },
  { id: "sec125", name: "Sector-125, Noida", lat: 28.5456, lng: 77.3261, areaType: "Institutional/Urban" },
  { id: "kp3", name: "Knowledge Park III, Greater Noida", lat: 28.4682, lng: 77.4912, areaType: "Educational/Sparsely Populated" },
  { id: "sec1", name: "Sector-1, Noida", lat: 28.5862, lng: 77.3094, areaType: "Commercial/Industrial Edge" }
];

export const ML_METRICS: ModelMetric[] = [
  { name: "HistGradientBoosting", type: "ML", rmse: 1.527, mae: 0.830, mape: 0.51, r2: 0.9998, trainTime: 3.4, inferenceTime: 0.12, memory: "42", isProduction: true },
  { name: "DecisionTree", type: "ML", rmse: 2.087, mae: 0.422, mape: 0.28, r2: 0.9996, trainTime: 1.1, inferenceTime: 0.05, memory: "22" },
  { name: "GradientBoosting", type: "ML", rmse: 5.289, mae: 3.604, mape: 2.02, r2: 0.9976, trainTime: 12.8, inferenceTime: 0.18, memory: "55" },
  { name: "RandomForest", type: "ML", rmse: 8.257, mae: 5.181, mape: 2.85, r2: 0.9943, trainTime: 28.4, inferenceTime: 1.45, memory: "180" },
  { name: "LinearRegression", type: "ML", rmse: 12.960, mae: 9.585, mape: 5.63, r2: 0.9859, trainTime: 1.5, inferenceTime: 0.08, memory: "12" },
  { name: "RidgeRegression", type: "ML", rmse: 12.974, mae: 9.592, mape: 5.64, r2: 0.9858, trainTime: 1.2, inferenceTime: 0.07, memory: "12" }
];

export const DL_METRICS: ModelMetric[] = [
  { name: "CNN-LSTM Hybrid", type: "DL", rmse: 22.174, mae: 15.603, mape: 7.73, r2: 0.9582, trainTime: 245.0, inferenceTime: 2.45, memory: "210", isProduction: true },
  { name: "LSTM (Long Short-Term Memory)", type: "DL", rmse: 22.653, mae: 15.854, mape: 7.73, r2: 0.9564, trainTime: 180.0, inferenceTime: 1.95, memory: "154" },
  { name: "GRU (Gated Recurrent Unit)", type: "DL", rmse: 22.827, mae: 16.484, mape: 8.02, r2: 0.9557, trainTime: 155.0, inferenceTime: 1.62, memory: "125" }
];

export const GLOBAL_SHAP: SHAPFeature[] = [
  { name: "CO × PM2.5 Interaction", value: 36.42, featureValue: "0.12 Importance" },
  { name: "PM2.5 Concentration", value: 36.39, featureValue: "0.12 Importance" },
  { name: "AQI (7-day lag)", value: 28.41, featureValue: "0.09 Importance" },
  { name: "AQI (24h lag)", value: 25.56, featureValue: "0.08 Importance" },
  { name: "AQI (7-day rolling mean)", value: 24.38, featureValue: "0.08 Importance" },
  { name: "PM2.5 (24h lag)", value: 17.85, featureValue: "0.06 Importance" },
  { name: "PM2.5 (7-day rolling mean)", value: 16.78, featureValue: "0.06 Importance" },
  { name: "PM2.5 × Humidity Interaction", value: 16.45, featureValue: "0.05 Importance" },
  { name: "CO Concentration", value: 14.90, featureValue: "0.05 Importance" },
  { name: "PM2.5 (7-day lag)", value: 11.94, featureValue: "0.04 Importance" }
];

export const MONTHLY_AQI_PATTERN = [
  { month: "Jan", pm25: 185, pm10: 290, no2: 58, o3: 25, aqi: 245 },
  { month: "Feb", pm25: 142, pm10: 220, no2: 45, o3: 32, aqi: 198 },
  { month: "Mar", pm25: 98, pm10: 165, no2: 32, o3: 45, aqi: 135 },
  { month: "Apr", pm25: 78, pm10: 155, no2: 28, o3: 54, aqi: 118 },
  { month: "May", pm25: 65, pm10: 175, no2: 24, o3: 65, aqi: 125 },
  { month: "Jun", pm25: 55, pm10: 160, no2: 20, o3: 58, aqi: 110 },
  { month: "Jul", pm25: 35, pm10: 85, no2: 15, o3: 28, aqi: 62 },
  { month: "Aug", pm25: 32, pm10: 78, no2: 12, o3: 24, aqi: 58 },
  { month: "Sep", pm25: 48, pm10: 112, no2: 22, o3: 35, aqi: 85 },
  { month: "Oct", pm25: 124, pm10: 235, no2: 48, o3: 42, aqi: 184 },
  { month: "Nov", pm25: 245, pm10: 380, no2: 72, o3: 30, aqi: 310 },
  { month: "Dec", pm25: 220, pm10: 340, no2: 65, o3: 22, aqi: 285 }
];

export const CORRELATION_MATRIX = {
  features: ["PM2.5", "PM10", "NO2", "O3", "Temp", "Humidity", "WindSpeed", "Rainfall"],
  matrix: [
    [1.00, 0.88, 0.65, -0.21, -0.45, 0.38, -0.52, -0.15], // PM2.5
    [0.88, 1.00, 0.58, -0.15, -0.38, 0.28, -0.48, -0.12], // PM10
    [0.65, 0.58, 1.00, -0.32, -0.28, 0.18, -0.35, -0.08], // NO2
    [-0.21, -0.15, -0.32, 1.00, 0.68, -0.55, 0.12, -0.05], // O3
    [-0.45, -0.38, -0.28, 0.68, 1.00, -0.62, 0.22, 0.14], // Temp
    [0.38, 0.28, 0.18, -0.55, -0.62, 1.00, -0.42, 0.25], // Humidity
    [-0.52, -0.48, -0.35, 0.12, 0.22, -0.42, 1.00, 0.08], // WindSpeed
    [-0.15, -0.12, -0.08, -0.05, 0.14, 0.25, 0.08, 1.00]  // Rainfall
  ]
};

// Generate 365 calendar grid squares representing Noyda winter heavy violations (GitHub heatmap grid)
export const generateCalendarData = (year: number) => {
  const data = [];
  const startDate = new Date(year, 0, 1);
  const endDate = new Date(year, 11, 31);
  const currentDate = new Date(startDate);

  while (currentDate <= endDate) {
    const month = currentDate.getMonth();
    const day = currentDate.getDate();
    // Heavy pollution values in Nov (10), Dec (11), Jan (0), Feb (1)
    let baseline = 60;
    if (month === 10 || month === 11) baseline = 220; // Smog peak
    else if (month === 0 || month === 1) baseline = 180;
    else if (month === 9) baseline = 140; // Autumn rise
    else if (month === 5 || month === 6 || month === 7) baseline = 45; // Monsoons

    // Add some random volatility
    const rand = Math.sin(day / 3) * 15 + (Math.random() * 20 - 10);
    const aqi = Math.max(15, Math.round(baseline + rand));

    data.push({
      date: currentDate.toISOString().split("T")[0],
      dayOfWeek: currentDate.getDay(),
      month: month,
      aqi
    });

    currentDate.setDate(currentDate.getDate() + 1);
  }
  return data;
};
