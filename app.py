from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/record', methods=['POST'])
def record():
    data = request.get_json()
    if data['start']:
        # Here you would add your logic to start recording
        return jsonify(message="Recording has started.")
    else:
        # Here you would add your logic to stop recording
        return jsonify(message="Recording is finished.")

if __name__ == '__main__':
    app.run(debug=True)
