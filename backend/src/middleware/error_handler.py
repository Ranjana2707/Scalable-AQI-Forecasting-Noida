from flask import jsonify

def setup_error_handlers(app):
    @app.errorhandler(ValueError)
    def handle_value_error(e):
        return jsonify({ "error": "Invalid Input Parameter", "details": str(e) }), 400
        
    @app.errorhandler(KeyError)
    def handle_key_error(e):
        return jsonify({ "error": "Missing Required Configuration Key", "details": str(e) }), 400
        
    @app.errorhandler(Exception)
    def handle_generic_exception(e):
        print(f"[ERROR EXCEPTION] {e}")
        return jsonify({ "error": "Internal Server Error", "details": str(e) }), 500
