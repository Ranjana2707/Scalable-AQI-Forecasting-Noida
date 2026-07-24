from flask import Blueprint, request, jsonify

last_active_model = "HistGradientBoosting"
last_active_station = "sec62"
last_active_date = "2026-06-26"
last_active_time = "12:00"

def create_shap_blueprint(shap_service, predict_service):
    bp = Blueprint("shap", __name__)
    
    @bp.route("/api/v1/shap", methods=["POST"])
    def shap():
        req_data = request.json or {}
        res = shap_service.calculate_shap(req_data, last_active_model, last_active_station, last_active_date)
        return jsonify(res)
        
    @bp.route("/api/v1/shap/predict", methods=["POST"])
    def shap_predict():
        req_data = request.json or {}
        pred_res = predict_service.run_prediction(req_data)
        shap_res = shap_service.calculate_shap(req_data, last_active_model, last_active_station, last_active_date)
        return jsonify({
            "prediction": pred_res,
            "shap": shap_res
        })
        
    @bp.route("/api/v1/shap/global", methods=["GET"])
    def shap_global():
        from src.config.app_config import FEAT_COLS
        import numpy as np
        model, model_type = shap_service.model_manager.get_model(last_active_model)
        global_importance = []
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            total = sum(importances) or 1.0
            ui_mapping = {
                "pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "so2": "SO2", "co": "CO", "o3": "O3",
                "wind_speed": "Wind Speed", "humidity": "Humidity", "temperature": "Temperature"
            }
            items = []
            for col, imp in zip(FEAT_COLS, importances):
                displayName = ui_mapping.get(col, col.replace("_", " ").title())
                items.append({ "feature": displayName, "importance": round(float(imp / total * 100), 2) })
            items.sort(key=lambda x: x["importance"], reverse=True)
            global_importance = items[:8]
        else:
            shap_bg = shap_service.dataset_repo.shap_bg
            from src.explainability.shap_explainer import KernelSHAPExplainer
            explainer = KernelSHAPExplainer(model, shap_bg[:20], feature_names=FEAT_COLS, seed=42)
            shap_vals = explainer.shap_values(shap_bg[:10], n_coalitions=32)
            mean_abs = np.mean(np.abs(shap_vals), axis=0)
            total = sum(mean_abs) or 1.0
            items = []
            for col, imp in zip(FEAT_COLS, mean_abs):
                items.append({ "feature": col.replace("_", " ").title(), "importance": round(float(imp / total * 100), 2) })
            items.sort(key=lambda x: x["importance"], reverse=True)
            global_importance = items[:8]
            
        return jsonify({ "globalFeatureImportance": global_importance })
        
    @bp.route("/api/v1/shap/local", methods=["GET"])
    def shap_local():
        req_data = {
            "modelName": last_active_model,
            "stationId": last_active_station,
            "date": last_active_date
        }
        res = shap_service.calculate_shap(req_data, last_active_model, last_active_station, last_active_date)
        return jsonify(res)
        
    @bp.route("/api/v1/shap/summary", methods=["GET"])
    def shap_summary():
        return jsonify({ "plotType": "summary", "data": shap_service.calculate_shap({}, last_active_model, last_active_station, last_active_date) })
        
    @bp.route("/api/v1/shap/beeswarm", methods=["GET"])
    def shap_beeswarm():
        return jsonify({ "plotType": "beeswarm", "data": shap_service.calculate_shap({}, last_active_model, last_active_station, last_active_date) })
        
    @bp.route("/api/v1/shap/waterfall", methods=["GET"])
    def shap_waterfall():
        return jsonify({ "plotType": "waterfall", "data": shap_service.calculate_shap({}, last_active_model, last_active_station, last_active_date) })
        
    @bp.route("/api/v1/shap/force", methods=["GET"])
    def shap_force():
        return jsonify({ "plotType": "force", "data": shap_service.calculate_shap({}, last_active_model, last_active_station, last_active_date) })
        
    @bp.route("/api/v1/shap/dependence", methods=["GET"])
    def shap_dependence():
        return jsonify({ "plotType": "dependence", "data": shap_service.calculate_shap({}, last_active_model, last_active_station, last_active_date) })
        
    @bp.route("/api/v1/shap/decision", methods=["GET"])
    def shap_decision():
        return jsonify({ "plotType": "decision", "data": shap_service.calculate_shap({}, last_active_model, last_active_station, last_active_date) })
        
    return bp
