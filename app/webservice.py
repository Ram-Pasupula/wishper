
import os
import logging
from os import path
from typing import Union
from fastapi import FastAPI, File, UploadFile, Query, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from .openai_whisper.core import transcriber
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
LANGUAGE_CODES = sorted(("en", "es"))

app = FastAPI(
    title="Contact Analytics Whisper API",
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
        task: Union[str, None] = Query(default="transcribe", enum=[
                                       "transcribe"]),
        lang: Union[str, None] = Query(default="en", enum=LANGUAGE_CODES),
        output: Union[str, None] = Query(
            default="txt", enum=["txt",  "json"])
):
    try:
        file_location = f"/tmp/{audio_file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(audio_file.file.read())
        file_path = os.path.dirname(file_location)
        logger.info(f"task : {task}")
        logger.info(f"lang : {lang}")
        result = transcriber(f"{file_path}/{audio_file.filename}", task, lang, output)
        #logger.info(result)
    except Exception:
        raise Exception(status_code=500, detail='File not able to load')
    else:
        return StreamingResponse(
            result,
            media_type="text/plain",
            headers={
                'Content-Disposition': f'attachment; filename="{audio_file.filename}.{output}"'
            })
    finally:
        try:
            import glob
            files = glob.glob(f"/tmp/{audio_file.filename}")
            for f in files:
                os.remove(f)
        except Exception:
            pass
        else:
            logger.info("Successfully deleted temp files")