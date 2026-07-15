export type Tab =
  | "dashboard"
  | "forecast"
  | "eda"
  | "shap"
  | "ml"
  | "dl"
  | "stations"
  | "seasonal"
  | "research";

export interface Pollutants {
  pm25: number;
  pm10: number;
  no2: number;
  so2: number;
  co: number;
  o3: number;
  nh3: number;
}

export interface Meteorology {
  temperature: number;
  humidity: number;
  windSpeed: number;
  rainfall: number;
}

export interface AqiCategoryInfo {
  label: string;
  color: string;
  bgColor: string;
  textColor: string;
  borderColor: string;
  description: string;
}

export interface ModelMetric {
  name: string;
  type: "ML" | "DL";
  rmse: number;
  mae: number;
  mape: number;
  r2: number;
  trainTime: number; // seconds
  inferenceTime: number; // milliseconds
  memory: string; // MB
  isProduction?: boolean;
}

export interface SHAPFeature {
  name: string;
  value: number; // impact on AQI
  featureValue: string; // e.g. "154 µg/m³" or "12 km/h"
}

export interface DataPoint {
  date: string;
  pm25: number;
  pm10: number;
  no2: number;
  o3: number;
  aqi: number;
}
