import threading
import queue
import re
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import soundfile as sf


HARDCODED_VOICES = [
    "p326",
    "p226",
    "p227",
    "p230",
    "p231",
    "p232",
    "p239",
    "p250",
    "p270",
    "p300",
]

DEFAULT_TTS_MODEL_NAME = "tts_models/en/vctk/vits"
OUTPUT_GAIN_DB = 9.0


class EchoCoquiApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("EchoCoqui")
        self.minsize(900, 650)

        self._worker_thread: threading.Thread | None = None
        self._worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self._models: list[str] = []

        self._build_ui()
        self._schedule_queue_pump()

        self._load_models_async()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        root = ttk.Frame(self, padding=12)
        root.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        top = ttk.Frame(root)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Voice / Model").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 8))

        self.model_var = tk.StringVar(value="")
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=0, column=1, sticky="ew", pady=(0, 8))

        self.refresh_button = ttk.Button(top, text="Refresh", command=self._load_models_async)
        self.refresh_button.grid(row=0, column=2, sticky="e", padx=(8, 0), pady=(0, 8))

        ttk.Label(top, text="Output file").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 8))

        self.output_var = tk.StringVar(value="")
        self.output_entry = ttk.Entry(top, textvariable=self.output_var)
        self.output_entry.grid(row=1, column=1, sticky="ew", pady=(0, 8))

        self.browse_button = ttk.Button(top, text="Browse...", command=self._browse_output)
        self.browse_button.grid(row=1, column=2, sticky="e", padx=(8, 0), pady=(0, 8))

        ttk.Label(root, text="Text").grid(row=1, column=0, sticky="w")

        text_frame = ttk.Frame(root)
        text_frame.grid(row=2, column=0, sticky="nsew", pady=(6, 10))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.text = tk.Text(text_frame, wrap="word", undo=True)
        self.text.grid(row=0, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.text.configure(yscrollcommand=scroll.set)

        bottom = ttk.Frame(root)
        bottom.grid(row=3, column=0, sticky="ew")
        bottom.columnconfigure(1, weight=1)

        self.start_button = ttk.Button(bottom, text="Start", command=self._start_generation)
        self.start_button.grid(row=0, column=0, sticky="w")

        self.progress = ttk.Progressbar(bottom, mode="indeterminate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=10)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var).grid(row=0, column=2, sticky="e")

    def _set_busy(self, busy: bool) -> None:
        if busy:
            self.start_button.configure(state="disabled")
            self.browse_button.configure(state="disabled")
            self.refresh_button.configure(state="disabled")
            self.model_combo.configure(state="disabled")
            self.output_entry.configure(state="disabled")
            self.text.configure(state="disabled")
            self.progress.start(10)
        else:
            self.start_button.configure(state="normal")
            self.browse_button.configure(state="normal")
            self.refresh_button.configure(state="normal")
            self.model_combo.configure(state="readonly")
            self.output_entry.configure(state="normal")
            self.text.configure(state="normal")
            self.progress.stop()

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Select output WAV file",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def _load_models_async(self) -> None:
        self._models = list(HARDCODED_VOICES)
        self._models.sort()
        self.model_combo.configure(values=self._models)
        if not self.model_var.get() and self._models:
            self.model_var.set(self._models[0])
        self.status_var.set("Ready")
        self._set_busy(False)

    def _start_generation(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return

        speaker_id = self.model_var.get().strip()
        if not speaker_id:
            messagebox.showerror("Missing model", "Please select a voice/model.")
            return

        output_path = self.output_var.get().strip()
        if not output_path:
            messagebox.showerror("Missing output", "Please choose an output file path.")
            return

        text = self.text.get("1.0", "end").strip()
        if not text:
            messagebox.showerror("Missing text", "Please enter some text.")
            return

        out = Path(output_path)
        if out.suffix.lower() != ".wav":
            out = out.with_suffix(".wav")
            self.output_var.set(str(out))

        self.status_var.set("Generating...")
        self._set_busy(True)

        def worker() -> None:
            try:
                from TTS.api import TTS

                tts = TTS(model_name=DEFAULT_TTS_MODEL_NAME, progress_bar=False, gpu=False)

                chunks = _chunk_text(text, max_chars=450)
                audio_parts: list[np.ndarray] = []

                sample_rate = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", None)
                if not sample_rate:
                    sample_rate = 22050

                silence = np.zeros(int(sample_rate * 0.15), dtype=np.float32)

                for i, chunk in enumerate(chunks, start=1):
                    self._worker_queue.put(("status", f"Synthesizing chunk {i}/{len(chunks)}"))
                    wav = tts.tts(chunk, speaker=speaker_id)
                    wav_np = np.asarray(wav, dtype=np.float32)
                    audio_parts.append(wav_np)
                    audio_parts.append(silence)

                if not audio_parts:
                    raise ValueError("No audio generated")

                final_audio = np.concatenate(audio_parts)
                gain = float(10 ** (OUTPUT_GAIN_DB / 20.0))
                target_rms = float(_rms(final_audio) * gain)

                boosted = final_audio * gain
                limited = np.tanh(boosted)

                limited_rms = float(_rms(limited))
                limited_peak = float(np.max(np.abs(limited))) if limited.size else 0.0

                if limited_rms > 0.0 and limited_peak > 0.0:
                    makeup = target_rms / limited_rms
                    max_makeup = 0.99 / limited_peak
                    final_audio = limited * min(makeup, max_makeup)
                else:
                    final_audio = limited
                sf.write(str(out), final_audio, int(sample_rate), subtype="PCM_16")

                self._worker_queue.put(("done", str(out)))
            except Exception as e:
                self._worker_queue.put(("error", e))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _schedule_queue_pump(self) -> None:
        self.after(100, self._pump_queue)

    def _pump_queue(self) -> None:
        try:
            while True:
                kind, payload = self._worker_queue.get_nowait()

                if kind == "models":
                    models = list(payload)  # type: ignore[arg-type]
                    models.sort()
                    self._models = models
                    self.model_combo.configure(values=self._models)
                    if not self.model_var.get() and self._models:
                        self.model_var.set(self._models[0])
                    self.status_var.set("Ready")
                    self._set_busy(False)

                elif kind == "status":
                    self.status_var.set(str(payload))

                elif kind == "done":
                    self.status_var.set("Done")
                    self._set_busy(False)
                    messagebox.showinfo("Generated", f"Audio saved to:\n{payload}")

                elif kind == "error":
                    self.status_var.set("Error")
                    self._set_busy(False)
                    messagebox.showerror("Error", str(payload))

        except queue.Empty:
            pass
        finally:
            self._schedule_queue_pump()


def _chunk_text(text: str, max_chars: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    parts = re.split(r"(?<=[.!?])\s+", cleaned)

    chunks: list[str] = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if len(part) > max_chars:
            for i in range(0, len(part), max_chars):
                sub = part[i : i + max_chars].strip()
                if sub:
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(sub)
            continue

        if not current:
            current = part
            continue

        if len(current) + 1 + len(part) <= max_chars:
            current = f"{current} {part}"
        else:
            chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    return chunks


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def main() -> None:
    app = EchoCoquiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
