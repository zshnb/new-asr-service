import string

from functools import cached_property
from typing import List, Optional, Tuple
from functools import lru_cache

import tokenizers


class Tokenizer:
    """Simple wrapper around a tokenizers.Tokenizer."""

    def __init__(
        self,
        tokenizer: tokenizers.Tokenizer,
        multilingual: bool,
        task: Optional[str] = None,
        language: Optional[str] = None,
        info_language: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.unicode_lang = set(_UNICODE_RANGE.keys())
        self.info_language = info_language


        if multilingual:
            if task not in _TASKS:
                raise ValueError(
                    "'%s' is not a valid task (accepted tasks: %s)"
                    % (task, ", ".join(_TASKS))
                )

            if language not in _LANGUAGE_CODES:
                raise ValueError(
                    "'%s' is not a valid language code (accepted language codes: %s)"
                    % (language, ", ".join(_LANGUAGE_CODES))
                )

            self.task = self.tokenizer.token_to_id("<|%s|>" % task)
            self.language = self.tokenizer.token_to_id("<|%s|>" % language)
            self.language_code = language
        else:
            self.task = None
            self.language = None
            self.language_code = "en"

    @cached_property
    def transcribe(self) -> int:
        return self.tokenizer.token_to_id("<|transcribe|>")

    @cached_property
    def translate(self) -> int:
        return self.tokenizer.token_to_id("<|translate|>")

    @cached_property
    def sot(self) -> int:
        return self.tokenizer.token_to_id("<|startoftranscript|>")

    @cached_property
    def sot_lm(self) -> int:
        return self.tokenizer.token_to_id("<|startoflm|>")

    @cached_property
    def sot_prev(self) -> int:
        return self.tokenizer.token_to_id("<|startofprev|>")

    @cached_property
    def eot(self) -> int:
        return self.tokenizer.token_to_id("<|endoftext|>")

    @cached_property
    def no_timestamps(self) -> int:
        return self.tokenizer.token_to_id("<|notimestamps|>")

    @property
    def timestamp_begin(self) -> int:
        return self.no_timestamps + 1

    @property
    def sot_sequence(self) -> List[int]:
        sequence = [self.sot]

        if self.language is not None:
            sequence.append(self.language)

        if self.task is not None:
            sequence.append(self.task)

        return sequence

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: List[int]) -> str:
        text_tokens = [token for token in tokens if token < self.eot]
        return self.tokenizer.decode(text_tokens)

    def decode_with_timestamps(self, tokens: List[int]) -> str:
        outputs = [[]]

        for token in tokens:
            if token >= self.timestamp_begin:
                timestamp = f"<|{(token - self.timestamp_begin) * 0.02:.2f}|>"
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)

        return "".join(
            [s if isinstance(s, str) else self.tokenizer.decode(s) for s in outputs]
        )

    @cached_property
    def non_speech_tokens(self) -> Tuple[int]:
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += (
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        )

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.encode(" -")[0], self.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [
                self.encode(symbol),
                self.encode(" " + symbol),
            ]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))

    def split_to_word_tokens(
        self, tokens: List[int]
    ) -> Tuple[List[str], List[List[int]]]:
        if self.language_code in {"zh", "ja", "th", "lo", "my", "yue"}:
            # These languages don't typically use spaces, so it is difficult to split words
            # without morpheme analysis. Here, we instead split words at any
            # position where the tokens are decoded as valid unicode points
            return self.split_tokens_on_unicode(tokens)
        elif self.info_language == 'multi':
            return self.split_tokens_on_multi(tokens)
        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(
        self, tokens: List[int]
    ) -> Tuple[List[str], List[List[int]]]:
        decoded_full = self.decode_with_timestamps(tokens)
        replacement_char = "\ufffd"

        words = []
        word_tokens = []
        current_tokens = []
        unicode_offset = 0

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)

            try:
                replacement_char_index = decoded.index(replacement_char)
                replacement_char_index += unicode_offset
            except ValueError:
                replacement_char_index = None

            if replacement_char_index is None or (
                replacement_char_index < len(decoded_full)
                and decoded_full[replacement_char_index] == replacement_char
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)

        return words, word_tokens

    def split_tokens_on_spaces(
        self, tokens: List[int]
    ) -> Tuple[List[str], List[List[int]]]:
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens
    
    def split_tokens_on_multi(
        self, tokens: List[int]
    ) -> Tuple[List[str], List[List[int]]]:
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            lang_cnt= {key: 0 for key in list(self.unicode_lang) + ['other']}
            pre_lang = None
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            for char in subword:
                lang_cnt[self.get_char_lang(ord(char))] += 1
            lang, _ = max(lang_cnt.items(), key=lambda x: x[1])
            if pre_lang:
                pre_lang = lang
            if special or with_space or punctuation or len(words) == 0 or lang in self.unicode_lang:
                words.append(subword)
                word_tokens.append(subword_tokens)
            elif lang not in self.unicode_lang and pre_lang in self.unicode_lang:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens

    @lru_cache(maxsize=12288)
    def get_char_lang(self, code):
        # CJK统一汉字
        if 0x4E00 <= code <= 0x9FFF:
            return 'zh'
        # CJK扩展区
        if 0x3400 <= code <= 0x4DBF:  # 扩展A
            return 'zh'
        if 0x20000 <= code <= 0x2A6DF:  # 扩展B
            return 'zh'
        if 0x2A700 <= code <= 0x2B73F:  # 扩展C
            return 'zh'
        if 0x2B740 <= code <= 0x2B81F:  # 扩展D
            return 'zh'
        # 日语
        if 0x3040 <= code <= 0x309F:  # 平假名
            return 'ja'
        if 0x30A0 <= code <= 0x30FF:  # 片假名
            return 'ja'
        if 0xFF66 <= code <= 0xFF9F:
            return 'ja'
        # 韩语
        if 0xAC00 <= code <= 0xD7AF:  # 谚文音节
            return 'ko'
        if 0x1100 <= code <= 0x11FF:  # 谚文字母
            return 'ko'
        # 泰语
        if 0x0E00 <= code <= 0x0E7F:
            return 'th'
        if 0x0E80 <= code <= 0x0EFF:
            return 'lo'
        if 0x1000 <= code <= 0x109F:
            return 'my'
        
        return 'other'


_TASKS = (
    "transcribe",
    "translate",
)

_UNICODE_RANGE = {
    "zh": [
        (0x4E00, 0x9FFF),   # CJK统一汉字
        (0x3400, 0x4DBF),   # CJK扩展A
        (0x20000, 0x2A6DF), # CJK扩展B
        (0x2A700, 0x2B73F), # CJK扩展C
        (0x2B740, 0x2B81F), # CJK扩展D
    ],
    "ja": [
        (0x4E00, 0x9FFF),   # CJK统一汉字
        (0x3040, 0x309F),   # 平假名
        (0x30A0, 0x30FF),   # 片假名
        (0xFF66, 0xFF9F),   # 半角片假名
    ],
    "ko": [
        (0xAC00, 0xD7AF),
        (0x1100, 0x11FF),
    ],
    "th": [(0x0E00, 0x0E7F)],  # 泰文
    "lo": [(0x0E80, 0x0EFF)],  # 老挝文
    "my": [(0x1000, 0x109F)],  # 缅甸文
}

_LANGUAGE_CODES = (
    "multi",
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
    "yue",
)
