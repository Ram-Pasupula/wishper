import os
from io import StringIO
from threading import Lock
from typing import Union, BinaryIO
import torch
import whisper
from .utils import ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON

# More about available quantization levels is here:
#   https://opennmt.net/CTranslate2/quantization.html

if torch.cuda.is_available():
    model = whisper.load_model("./large/large-v3.pt") .cuda()
else:
    model = whisper.load_model("./large/large-v3.pt")
model_lock = Lock()


def transcribe(
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        output,
):
    options_dict = {"task": task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    if word_timestamps:
        options_dict["vad_filter"] = True
    if word_timestamps:
        options_dict["word_timestamps"] = True
    with model_lock:
        segments = []
        text = ""
        segment_generator, info = model.transcribe(
            audio, beam_size=5, **options_dict)
        for segment in segment_generator:
            segments.append(segment)
            text = text + segment.text
        result = {
            "language": options_dict.get("language", info.language),
            "segments": segments,
            "text": text
        }

    output_file = StringIO()
    write_result(result, output_file, output)
    output_file.seek(0)

    return output_file


def language_detection(audio):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.pad_or_trim(audio)

    # detect the spoken language
    with model_lock:
        segments, info = model.transcribe(audio, beam_size=5)
        detected_lang_code = info.language
    return detected_lang_code


def write_result(
        result: dict, file: BinaryIO, output: Union[str, None]
):
    if output == "srt":
        WriteSRT(ResultWriter).write_result(result, file=file)
    elif output == "vtt":
        WriteVTT(ResultWriter).write_result(result, file=file)
    elif output == "tsv":
        WriteTSV(ResultWriter).write_result(result, file=file)
    elif output == "json":
        WriteJSON(ResultWriter).write_result(result, file=file)
    elif output == "txt":
        WriteTXT(ResultWriter).write_result(result, file=file)
    else:
        return 'Please select an output method!'
