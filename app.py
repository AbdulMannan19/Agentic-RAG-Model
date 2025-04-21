from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/rag', methods=['POST'])
def get_query_result():
    data = request.get_json()
    query = data.get('query', '')
    
    


    return jsonify({
        "query result": "Revenue in the financial year 24-25 is 1000000"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
