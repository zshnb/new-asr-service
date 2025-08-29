import logging
from dataclasses import dataclass
import torch
from lib.faster_whisper import WhisperModel
from util import timing


@dataclass
class TranscribeOption:
    beam_size: int
    hotwords: str
    vad_filter: bool
    vad_parameters: dict
    word_timestamps_dict: dict

class Transcriber:
    def __init__(self, model_size: str = 'large-v3-turbo', num_workers: int = 2):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        compute_type = 'float16' if torch.cuda.is_available() else 'int8'
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type, num_workers=num_workers)
        self.initial_prompt = {
            'zh': '以下内容是一段中文对话，话题涉及金融、历史、日常生活、体育、自我提升等',
            'en': 'The follow is a conversation which include finance, history, daily life, sports, self-improvement etc.'
        }
        self.log_prob_low_threshold = -0.7

    @timing
    def transcribe_segment(self, segment_file: str, offset: int, options: TranscribeOption):
        logging.info(f'transcribe_segment: {segment_file} with options: {options}')
        segments, info = self.model.transcribe(
            segment_file,
            beam_size=options.beam_size,
            hotwords=options.hotwords,
            vad_filter=options.vad_filter,
            initial_prompt=self.initial_prompt,
            vad_parameters=options.vad_parameters,
            word_timestamps_dict=options.word_timestamps_dict,
            log_prob_low_threshold=self.log_prob_low_threshold,
        )

        results = []
        for segment in segments:
            results.append("[%.2fs -> %.2fs] %s" % (segment.start + offset, segment.end + offset, segment.text))
            # seg_word = []
            # for word in segment_file.words:
            #     seg_word.append({'start': word.start, 'end': word.end, 'word': word.word})
            # results.append(
            #     {
            #         'words': seg_word,
            #         'lang': info.language,
            #         'lang_prob': info.language_probability,
            #         'duration': info.duration,
            #         'duration_vad': info.duration_after_vad,
            #     }
            # )

        return results
