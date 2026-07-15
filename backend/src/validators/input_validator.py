import pandas as pd

class InputValidator:
    @staticmethod
    def validate_predict_payload(data):
        pollutants = data.get("pollutants", {})
        meteorology = data.get("meteorology", {})
        
        # Check presence of required parameters
        for key in ["pm25", "pm10", "no2"]:
            if key not in pollutants:
                raise ValueError(f"Missing pollutant: {key}")
            try:
                val = float(pollutants[key])
                if not (0.0 <= val <= 1000.0):
                    raise ValueError(f"Pollutant {key} value {val} out of valid bounds [0, 1000]")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Pollutant {key} must be a valid float value: {e}")
                
        for key in ["temperature", "humidity", "windSpeed"]:
            if key not in meteorology:
                raise ValueError(f"Missing meteorological covariate: {key}")
            try:
                val = float(meteorology[key])
                if key == "temperature" and not (-50.0 <= val <= 60.0):
                    raise ValueError(f"Temperature value {val} out of bounds [-50, 60]")
                elif key == "humidity" and not (0.0 <= val <= 100.0):
                    raise ValueError(f"Humidity value {val} out of bounds [0, 100]")
                elif key == "windSpeed" and not (0.0 <= val <= 250.0):
                    raise ValueError(f"Wind speed value {val} out of bounds [0, 250]")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Meteorological value {key} must be a valid float: {e}")
                
        # Validate station whitelist
        station_id = data.get("stationId", "sec62")
        if station_id not in ["sec62", "sec1", "kp3", "sec125"]:
            raise ValueError(f"Unauthorized or unknown station: {station_id}")
            
        # Validate model whitelist
        model_name = data.get("modelName", "HistGradientBoosting")
        allowed_models = ["HistGradientBoosting", "DecisionTree", "LSTM", "GRU", "CNN-LSTM"]
        if model_name not in allowed_models:
            raise ValueError(f"Unauthorized or unknown model: {model_name}")
            
        # Date & Time structure check to prevent Injection
        date_str = data.get("date")
        if date_str:
            if len(date_str) != 10:
                raise ValueError("Invalid date format length. Expected YYYY-MM-DD")
            try:
                pd.to_datetime(date_str, format="%Y-%m-%d")
            except Exception:
                raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")
                
        time_str = data.get("time")
        if time_str:
            if len(time_str) != 5 or ":" not in time_str:
                raise ValueError("Invalid time format. Expected HH:MM")
            try:
                parts = time_str.split(":")
                h, m = int(parts[0]), int(parts[1])
                if not (0 <= h <= 23 and 0 <= m <= 59):
                    raise ValueError("Time values out of valid hourly/minute range")
            except Exception:
                raise ValueError("Malformed time value")
                
        return True
