# EchoCoqui
Neural text-to-speech utility in Python using Coqui TTS. Generate high-quality speech audio offline from text to audio file.

## Requirements
Python 3.11.x (Coqui TTS compatible).

## Setup
Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
python echocoqui.py
```

## eSpeak NG (Windows)
Some Coqui TTS models require an eSpeak backend for phonemization. If you see:

`No espeak backend found. Install espeak-ng or espeak to your system.`

Install **eSpeak NG** and ensure it is on your `PATH`.

1. Install eSpeak NG for Windows.
2. Add the folder containing `espeak-ng.exe` to your user/system `PATH`.
   - Common locations are similar to:
     - `C:\Program Files\eSpeak NG\`
     - `C:\Program Files\eSpeak NG\bin\`

Verify in a new terminal:

```bash
espeak-ng --version
```

## Notes
- The model list is loaded from `TTS.list_models()`.
- Long text is chunked automatically to reduce synthesis failures.
