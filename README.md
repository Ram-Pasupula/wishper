![Release](https://img.shields.io/github/v/release/ahmetoner/whisper-asr-webservice.svg)
![Docker Pulls](https://img.shields.io/docker/pulls/onerahmet/openai-whisper-asr-webservice.svg)
<!-- ![Build](https://img.shields.io/github/actions/workflow/status/ahmetoner/whisper-asr-webservice/docker-publish.yml.svg) -->
![Licence](https://img.shields.io/github/license/ahmetoner/whisper-asr-webservice.svg)
# Whisper API Service

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitask model that can perform multilingual speech recognition as well as speech translation and language identification. For more details: [github.com/openai/whisper](https://github.com/openai/whisper/)

## Features
Current release (v1.0) supports following whisper models:

- transcribe
- detect-language


## Quick Usage

### CPU

```sh
docker run -d -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice:latest
```

### GPU

```sh
docker run -d --gpus all -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice:latest-gpu
```

for more information:

- [Documentation/Run](https://github.com/Ram-Pasupula/wishper.git)


# Documentation
Explore the documentation by clicking [here](https://github.com/Ram-Pasupula/wishper.git).
