import os
import json
from io import StringIO
from threading import Lock
from typing import BinaryIO, Union
import logging
import torch
from transformers import pipeline
from whisper.utils import ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_name = "base.pt"

if torch.cuda.is_available():
    device = "cuda"
    model_quantization = os.getenv("ASR_QUANTIZATION", "float32")
else:
    device = "cpu"
    model_quantization = os.getenv("ASR_QUANTIZATION", "int8")

MODEL_PATH_W = "./whisper"
whisper_m = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH_W,
    # device=device,
)
# for file in os.listdir("./large/"):
#     if file.endswith(".pt"):
#         file_name = file
# print(file_name)
mel = 128
# if not file_name.startswith("large"):
#     mel = 80
# if torch.cuda.is_available():
#     model = whisper.load_model(f"./large/{file_name}") .cuda()
# else:
#     model = whisper.load_model(f"./large/{file_name}")
model_lock = Lock()


def transcribe(
        audio,
        output
):

    with model_lock:
        result = whisper_m(audio,
                           chunk_length_s=30,
                           stride_length_s=5,
                           batch_size=8)
        print(result)
    output_file = StringIO()
    # WriteTXT(ResultWriter).write_result(result, file=output_file)
    # output_file = StringIO()
    # write_result(result, output_file, output)
    if "json" in output:
        json.dump(result["text"], output_file)
    else:
        print(result["text"], file=output_file, flush=True)
    output_file.seek(0)
    return output_file


# def language_detection(audio):
#     # load audio and pad/trim it to fit 30 seconds
#     audio = whisper.pad_or_trim(audio)
#     # make log-Mel spectrogram and move to the same device as the model
#     mels = whisper.log_mel_spectrogram(audio, n_mels=mel).to(model.device)
#     # detect the spoken language
#     with model_lock:
#         _, probs = model.detect_language(mels)
#         print("detect_language done")
#     detected_lang_code = max(probs, key=probs.get)
#     return detected_lang_code


def write_result(
        result: dict, file: BinaryIO, output: Union[str, None]
):
    options = {
        'max_line_width': 1000,
        'max_line_count': 10,
        'highlight_words': False
    }
    if output == "srt":
        WriteSRT(ResultWriter).write_result(result, file=file, options=options)
    elif output == "vtt":
        WriteVTT(ResultWriter).write_result(result, file=file, options=options)
    elif output == "tsv":
        WriteTSV(ResultWriter).write_result(result, file=file, options=options)
    elif output == "json":
        WriteJSON(ResultWriter).write_result(
            result, file=file, options=options)
    elif output == "txt":
        WriteTXT(ResultWriter).write_result(result, file=file, options=options)
    else:
        return 'Please select an output method!'
