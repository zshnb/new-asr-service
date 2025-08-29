import os
import subprocess
import logging


class RequestData:
    overlap_seconds = None
    segment_duration_seconds = None
    audio_file_path = None

    def __init__(self):
        self.overlap_seconds = None
        self.segment_duration_seconds = None
        self.audio_file_path = None

    def parse_from_request_json(self, request_json):
        self.audio_file_path = request_json['audio_file_path']
        self.segment_duration_seconds = request_json['segment_duration_seconds']
        self.overlap_seconds = request_json['overlap_seconds']


def create_clip(raw_audio: str, slice_audio: str, clipFromSecond: int, clipDuration: int):
    cmd_str = f"ffmpeg -y -ss {clipFromSecond} -t {clipDuration} -i {raw_audio} -vn {slice_audio} > /dev/null 2>&1"
    logging.info(f"[create_clip] command: {cmd_str}")

    try:
        os.system(cmd_str)
    except Exception as e:
        logging.error(f"[error on running ffmpeg]" ,e)
        os.system(f"cat {slice_audio}.log")
        raise e


def get_audio_file_length(file_path: str):
    cmd_str = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {file_path}"
    logging.info(f"[get_audio_file_length] command: {cmd_str}")

    try:
        output = subprocess.check_output(cmd_str, shell=True)
        logging.info(f"[get_audio_file_length] output: {output}")
        return int(float(output))
    except Exception as e:
        print(f"[error on running ffmpeg]")
        raise e


def split_audio_file_into_segments(audio_file, duration, segment_duration_seconds, overlap_seconds):
    logging.info(
        f"Splitting audio file {audio_file} into segments of {segment_duration_seconds} seconds each (total: {duration} seconds)")
    segments = []
    index = 0
    audio_dir = os.path.dirname(audio_file)
    if os.path.exists(f'{audio_dir}/segments'):
        return [f'{audio_dir}/segments/{x}' for x in os.listdir(f'{audio_dir}/segments')]
    os.mkdir(f"{audio_dir}/segments")
    audio_suffix = os.path.splitext(audio_file)[1] # already contains .(dot)
    merge_last_two_segments = False
    if duration % segment_duration_seconds < 100:
        merge_last_two_segments = True

    for i in range(0, duration, segment_duration_seconds):

        if merge_last_two_segments and i + 2 * segment_duration_seconds > duration > i + segment_duration_seconds:
            logging.info(f"Merging last two segments because the last segment is too short")
            segment_file = f"{audio_dir}/segments/{index}{audio_suffix}"
            create_clip(audio_file, segment_file, i, segment_duration_seconds + 100)
            segments.append(segment_file)
            break

        logging.info(f"Creating segment {i} to {i + segment_duration_seconds} seconds")
        segment_file = f"{audio_dir}/segments/{index}{audio_suffix}"
        create_clip(audio_file, segment_file, i, segment_duration_seconds + overlap_seconds)
        segments.append(segment_file)
        index += 1
    return segments


def run(args: RequestData):
    audio_file_path = args.audio_file_path
    duration = get_audio_file_length(audio_file_path)
    return split_audio_file_into_segments(audio_file_path, duration, args.segment_duration_seconds, args.overlap_seconds)
