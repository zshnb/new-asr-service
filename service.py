import logging
import os
import subprocess
import uuid
from split_audio_files import run as split_audio, RequestData


def handle_asr_task(audio_url: str):
    def download_audio():
        audio_file = f'tmp/{uuid.uuid4()}/input.mp3'
        os.mkdir(os.path.dirname(audio_file))
        subprocess.run(['ffmpeg', '-i', audio_url, audio_file])
        logging.info(f"[download_audio] audio_file: {audio_file}")
        return audio_file

    audio_file = download_audio()
    request_data = RequestData()
    request_data.parse_from_request_json({
        'audio_file_path': audio_file,
        'segment_duration_seconds': 600,
        'overlap_seconds': 0
    })

    split_audio(request_data)
