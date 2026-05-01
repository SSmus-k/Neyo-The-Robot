import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import pyttsx3
import os
from openai import OpenAI


API_KEY = "sk-c58edd3554bc4dc2a05079f51f550198p"
SAMPLE_RATE = 16000
BLOCK_DURATION = 1  


client = OpenAI(api_key=API_KEY)
model = whisper.load_model("tiny")
engine = pyttsx3.init()

audio_queue = queue.Queue()

def speak(text):
    print("neyo:", text)
    engine.say(text)
    engine.runAndWait()


def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


def process_audio():
    buffer = []

    while True:
        data = audio_queue.get()
        buffer.append(data)

       
        audio_data = np.concatenate(buffer, axis=0)

       
        if len(audio_data) > SAMPLE_RATE * 2:
            # Save temp file
            from scipy.io.wavfile import write
            write("temp.wav", SAMPLE_RATE, audio_data)

            
            result = model.transcribe("temp.wav")
            text = result["text"].strip().lower()

            if text:
                print("You:", text)

                if "stop" in text:
                    speak("Goodbye")
                    os._exit(0)

                if not control_system(text):
                    reply = ask_ai(text)
                    speak(reply)

            buffer.clear()  


def ask_ai(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def control_system(command):
    if "open chrome" in command:
        os.system("start chrome")
        speak("Opening Chrome")
    elif "open notepad" in command:
        os.system("notepad")
        speak("Opening Notepad")
    elif "shutdown" in command:
        speak("Shutting down system")
        os.system("shutdown /s /t 1")
    else:
        return False
    return True


def run_neyo():
    speak("neyo is now online")

   
    threading.Thread(target=process_audio, daemon=True).start()

   
    with sd.InputStream(callback=audio_callback,
                        channels=1,
                        samplerate=SAMPLE_RATE,
                        blocksize=int(SAMPLE_RATE * BLOCK_DURATION)):
        while True:
            pass  


run_neyo()