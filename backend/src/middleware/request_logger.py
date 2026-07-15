import time
from flask import request

def setup_logger(app):
    @app.before_request
    def before_request():
        request.start_time = time.time()
        
    @app.after_request
    def after_request(response):
        duration = time.time() - request.start_time
        print(f"[API Audit] {request.method} {request.path} | Status: {response.status_code} | Duration: {duration*1000:.1f}ms")
        return response
