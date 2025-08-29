from dataclasses import dataclass

from flask import Flask, request, jsonify
import datetime
import logging

app = Flask(__name__)

@dataclass
class CreateAsrRequest:
    audio_url: str
    @classmethod
    def from_json(cls, data: dict):
        return cls(audio_url=data['audio_url'])

@app.route('/api/asr/create', methods=['POST'])
def create_asr():
    try:
        data = CreateAsrRequest.from_json(request.get_json())
        response = {
            'message': '数据接收成功',
            'received_data': data,
            'processed_at': datetime.datetime.now().isoformat()
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
    logging.info('asr service running at localhost:8080')