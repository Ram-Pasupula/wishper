from transformers import pipeline
import os
import logging
from os import path
from typing import Union, Annotated
from fastapi import FastAPI, File, UploadFile, Query, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer
from .utility import load_audio
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
ASR_ENGINE = os.getenv("ASR_ENGINE", "openai_whisper")

if ASR_ENGINE == "faster_whisper":
    from .faster_whisper.core import transcribe, language_detection
else:
    from .openai_whisper.core import transcribe, language_detection

MODELS_PATH = "./summarize"
MODEL_PATH_TR = "./translate"
MODEL_PATH_W ="./whisper"
whisper = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH_W,
    chunk_length_s=30,
    #device=device,
)
summarize = pipeline(
    "summarization",
    model=MODELS_PATH,
)
translator = pipeline(
    "translation",
    model=MODEL_PATH_TR,
)

# LANGUAGE_CODES = sorted(list(tokenizer.LANGUAGES.keys()))
LANGUAGE_CODES = sorted(("en", "es"))
# projectMetadata = importlib.metadata.metadata('whisper-asr-webservice')
app = FastAPI(
    title="Contact Analytics AI API",
    # description="AI api is a general-purpose speech recognition api service.",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    redoc_url=None
)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")

    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )
    applications.get_swagger_ui_html = swagger_monkey_patch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/asr", tags=["Endpoints"])
async def asr(
        audio_file: UploadFile = File(...),
        encode: bool = Query(
            default=True, description="Encode audio first through ffmpeg"),
        task: Union[str, None] = Query(default="transcribe", enum=[
                                       "transcribe", "translate"]),
        language: Union[str, None] = Query(default="en", enum=LANGUAGE_CODES),
        initial_prompt: Union[str, None] = Query(default=None),
        vad_filter: Annotated[bool | None, Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if ASR_ENGINE ==
                               "faster_whisper" else False)
        )] = False,
        word_timestamps: bool = Query(
            default=False, description="Word level timestamps"),
        output: Union[str, None] = Query(
            default="txt", enum=["txt", "vtt", "srt", "tsv", "json"])
):
    result = transcribe(load_audio(audio_file.file, encode), task,
                        language, initial_prompt, vad_filter, word_timestamps, output)
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            'Asr-Engine': ASR_ENGINE,
            'Content-Disposition': f'attachment; filename="{audio_file.filename}.{output}"'
        })


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
        audio_file: UploadFile = File(...),
        encode: bool = Query(
            default=True, description="Encode audio first through ffmpeg")
):
    detected_lang_code = language_detection(
        load_audio(audio_file.file, encode))
    return {"detected_language": tokenizer.LANGUAGES[detected_lang_code], "language_code": detected_lang_code}


@app.post("/summarize", tags=["Endpoints"])
async def summarizer(article, max_length=400):
    sum_txt = summarize(article, max_length=int(max_length),
                        min_length=30, do_sample=False)
    return {"summarization": sum_txt}


@app.post("/translate", tags=["Endpoints"])
async def translator_es_en(article, max_length=400, src_lang="en", tgt_lang="en"):
    trans = translator(article, max_length=int(
        max_length),  src_lang=src_lang, tgt_lang=tgt_lang)
    return {"translation": trans}
