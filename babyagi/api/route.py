from flask import Blueprint, jsonify, request
from babyagi.functionz.core.framework import func

api = Blueprint('student_api', __name__)

@api.route('/api/functions', methods=['GET'])
def list_functions():
    functions = func.get_all_functions()
    return jsonify(functions)

@api.route('/api/functions/<function_name>', methods=['POST'])
def run_task(function_name):
    data = request.get_json()
    try:
        result = func.execute_function(function_name, **data)
        return jsonify({"output": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/api/logs/<function_name>', methods=['GET'])
def get_logs(function_name):
    logs = func.get_logs(function_name=function_name)
    return jsonify(logs)
