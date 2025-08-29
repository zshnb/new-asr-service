import argparse
import datetime
import hashlib
import logging
import os
import subprocess
from concurrent.futures.thread import ThreadPoolExecutor
from math import floor

from split_audio_files import run as split_audio, RequestData
from transcriber import Transcriber, TranscribeOption
from util import timing


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
@timing
def handle_asr_task(audio_url: str, num_workers: int, segment_duration: int):
    def download_audio():
        logging.info(f"[download_audio] audio_url: {audio_url}")
        md5 = hashlib.md5(audio_url.encode()).hexdigest()
        audio_file = f'tmp/{md5}/input.mp3'
        if os.path.exists(audio_file):
            logging.info(f"[download_audio] audio url {audio_url} already exists, skip download")
            return audio_file
        os.makedirs(os.path.dirname(audio_file), exist_ok=True)
        subprocess.run(['ffmpeg', '-i', audio_url, audio_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info(f"[download_audio] audio_file: {audio_file}")
        return audio_file

    def submit_all_transcription_tasks():
        def sort_and_concat():
            result = []
            keys = []
            for item_dict in tasks_results:
                def extract_number(filename):
                    import re
                    match = re.search(r'(\d+)', filename)
                    return int(match.group(1)) if match else 0

                keys.append(extract_number(item_dict['segment']))

            keys = sorted(keys)

            for key in keys:
                filtered_dict = list(filter(lambda x: x['segment'] == f'{key}.mp3', tasks_results))[0]
                result.append(filtered_dict['transcription_result'])

            return result

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            logging.info(f"[submit_all_transcription_tasks] executing tasks count: {len(tasks)}")
            futures = [executor.submit(task) for task in tasks]
            tasks_results = [future.result() for future in futures]

        return sort_and_concat()
    def do_transcription(segment_file: str):
        index = int(os.path.basename(segment_file).split('.')[0])
        return {
            'segment': os.path.basename(segment_file),
            'transcription_result': transcriber.transcribe_segment(segment_file, segment_duration * index, transcribe_option)
        }

    audio_file = download_audio()
    request_data = RequestData()
    request_data.parse_from_request_json({
        'audio_file_path': audio_file,
        'segment_duration_seconds': segment_duration,
        'overlap_seconds': 0
    })

    audio_segments = split_audio(request_data)
    transcriber = Transcriber(num_workers=num_workers)
    transcribe_option = TranscribeOption(5, "", True, {
        'onset': 0.6,
        'offset': 0.4,
        'min_silence_duration_ms': 500,
        'speech_pad_ms': 0,
        'min_speech_duration_ms': 160,
    }, {'zh': True, 'default': False})

    tasks = [lambda segment=segment: do_transcription(segment) for segment in audio_segments]
    return submit_all_transcription_tasks()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="asr service arguments")
    parser.add_argument("--num_workers", type=int, default=6, help="parallel worker count")
    parser.add_argument("--segment_duration", type=int, default=600, help="duration in seconds of each segment")
    parser.add_argument("--model_size", type=str, default='large-v3-turbo', help="model")
    parser.add_argument("--audio_url", type=str, help="Audio url")
    args = parser.parse_args()

    result = handle_asr_task(args.audio_url, args.num_workers, args.segment_duration)
    os.makedirs("test_data", exist_ok=True)
    output_file = f"test_data/{floor(datetime.datetime.now().timestamp())}_{args.num_workers}_{args.segment_duration}.txt"
    with open(output_file, 'w') as f:
        for item in result:
            for i in item:
                f.write(f"{i}\n")
        logging.info(f"write asr result: {output_file}")