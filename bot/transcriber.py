import os
import queue
import threading
from datetime import datetime

import numpy as np
import sounddevice as sd
import tkinter as tk
import whisper


SAMPLE_RATE = 16000
CHUNK_SECONDS = 3
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

audio_queue = queue.Queue()
is_recording = False


def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())


def transcribe_chunk(audio_data, mdl):
    # convert int16 -> float32 in [-1, 1] and flatten to 1D
    # whisper accepts numpy arrays directly, no ffmpeg needed
    audio_float = audio_data.flatten().astype(np.float32) / 32768.0
    result = mdl.transcribe(audio_float, fp16=False)
    return result["text"].strip()


def save_transcript(text):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(OUTPUT_DIR, f"transcript_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename


def recording_loop(mdl, on_partial, on_done):
    global is_recording
    buffer = []
    full_text = []
    chunk_samples = SAMPLE_RATE * CHUNK_SECONDS

    with sd.InputStream(callback=audio_callback, channels=1,
                        samplerate=SAMPLE_RATE, dtype="int16"):
        while is_recording:
            try:
                chunk = audio_queue.get(timeout=0.5)
                buffer.append(chunk)

                total = sum(len(c) for c in buffer)
                if total >= chunk_samples:
                    audio = np.concatenate(buffer, axis=0)
                    buffer.clear()
                    text = transcribe_chunk(audio, mdl)
                    if text:
                        full_text.append(text)
                        on_partial(" ".join(full_text))

            except queue.Empty:
                continue

    if buffer:
        audio = np.concatenate(buffer, axis=0)
        text = transcribe_chunk(audio, mdl)
        if text:
            full_text.append(text)

    final = " ".join(full_text)
    filepath = save_transcript(final) if final else None
    on_done(final, filepath)


# ── GUI ──────────────────────────────────────────────────────────────────────

BG   = "#0f0f0f"
BG2  = "#171717"
FG   = "#e8e8e8"
DIM  = "#555555"
RED  = "#ee0055"
FONT = ("Courier", 12)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("transcriber")
        self.root.configure(bg=BG)
        self.root.geometry("660x500")
        self.root.resizable(True, True)

        self.model = None
        self._thread = None
        self._blink_on = False

        self._build()
        self._load_model()

    def _build(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=20, pady=(18, 0))

        tk.Label(header, text="transcriber", font=("Courier", 14),
                 fg=DIM, bg=BG).pack(side="left")

        self.dot = tk.Label(header, text="●", font=("Courier", 14),
                            fg="#333", bg=BG)
        self.dot.pack(side="right", padx=(6, 0))

        self.status = tk.Label(header, text="loading model...",
                               font=FONT, fg=DIM, bg=BG)
        self.status.pack(side="right")

        self.text = tk.Text(
            self.root,
            font=("Courier", 12),
            fg=FG, bg=BG2,
            relief="flat",
            wrap="word",
            padx=14, pady=14,
            spacing3=4,
            state="disabled",
            cursor="arrow",
        )
        self.text.pack(fill="both", expand=True, padx=20, pady=14)

        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="x", padx=20, pady=(0, 18))

        self.save_label = tk.Label(bottom, text="", font=("Courier", 9),
                                   fg=DIM, bg=BG)
        self.save_label.pack(side="left", anchor="w")

        self.btn = tk.Button(
            bottom,
            text="● record",
            font=("Courier", 12),
            fg=FG, bg="#1e1e1e",
            activebackground="#2a2a2a",
            activeforeground=FG,
            relief="flat",
            padx=14, pady=7,
            cursor="hand2",
            state="disabled",
            command=self.toggle,
        )
        self.btn.pack(side="right")

        self._blink_loop()

    def _load_model(self):
        def load():
            mdl = whisper.load_model("tiny")
            self.root.after(0, self._model_ready, mdl)
        threading.Thread(target=load, daemon=True).start()

    def _model_ready(self, mdl):
        self.model = mdl
        self.status.config(text="idle", fg=DIM)
        self.btn.config(state="normal")

    def _blink_loop(self):
        if is_recording:
            color = RED if self._blink_on else "#660022"
            self.dot.config(fg=color)
            self._blink_on = not self._blink_on
        else:
            self.dot.config(fg="#333")
        self.root.after(500, self._blink_loop)

    def toggle(self):
        global is_recording
        if not is_recording:
            self._start()
        else:
            self._stop()

    def _start(self):
        global is_recording
        is_recording = True
        self._set_text("")
        self.save_label.config(text="")
        self.btn.config(text="■ stop", fg=RED)
        self.status.config(text="listening...", fg=RED)
        self._thread = threading.Thread(
            target=recording_loop,
            args=(self.model, self._on_partial, self._on_done),
            daemon=True,
        )
        self._thread.start()

    def _stop(self):
        global is_recording
        is_recording = False
        self.btn.config(state="disabled", text="● record", fg=FG)
        self.status.config(text="processing...", fg=DIM)

    def _on_partial(self, text):
        self.root.after(0, self._set_text, text)

    def _on_done(self, text, filepath):
        def update():
            self._set_text(text if text else "(nothing detected)")
            self.btn.config(state="normal")
            self.status.config(text="idle", fg=DIM)
            if filepath:
                self.save_label.config(
                    text=f"saved → {os.path.basename(filepath)}", fg=DIM)
        self.root.after(0, update)

    def _set_text(self, text):
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", text)
        self.text.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()