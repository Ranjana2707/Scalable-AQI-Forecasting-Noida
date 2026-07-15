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
        global_importance = [
            { "feature": "PM2.5", "importance": 45.62 },
            { "feature": "PM10", "importance": 24.15 },
            { "feature": "NO2", "importance": 12.80 },
            { "feature": "Wind Speed", "importance": 18.34 },
            { "feature": "Humidity", "importance": 9.45 },
            { "feature": "Temperature", "importance": 7.21 },
            { "feature": "SO2", "importance": 5.12 },
            { "feature": "CO", "importance": 4.88 }
        ]
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
