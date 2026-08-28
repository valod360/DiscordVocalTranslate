import io
import wave

import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel


DEVICE_INDEX = 23
RECORD_SECONDS = 4
CHUNK = 1024


print("Chargement de Whisper...")

try:
    model = WhisperModel(
        "turbo",
        device="cuda",
        compute_type="float16"
    )
    print("Whisper chargé sur le GPU ✅")

except Exception as e:
    print("GPU indisponible pour Whisper :", e)
    print("Passage sur CPU...")

    model = WhisperModel(
        "turbo",
        device="cpu",
        compute_type="int8"
    )


p = pyaudio.PyAudio()

device = p.get_device_info_by_index(DEVICE_INDEX)

RATE = int(device["defaultSampleRate"])
CHANNELS = int(device["maxInputChannels"])

print()
print("Périphérique :", device["name"])
print("Fréquence :", RATE)
print("Canaux :", CHANNELS)
print()
print("Écoute en cours...")
print("Parle espagnol sur Discord.")
print("Ctrl+C pour arrêter.")
print()


stream = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK,
)


try:

    while True:

        frames = []

        print("🎧 Écoute...", end="", flush=True)

        for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):

            data = stream.read(
                CHUNK,
                exception_on_overflow=False
            )

            frames.append(data)

        print("\r🧠 Transcription...", end="", flush=True)

        # Création d'un WAV directement en mémoire
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wf:

            wf.setnchannels(CHANNELS)
            wf.setsampwidth(
                p.get_sample_size(pyaudio.paInt16)
            )
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))

        wav_buffer.seek(0)

        segments, info = model.transcribe(
            wav_buffer,

            language="es",

            beam_size=1,

            vad_filter=True,

            vad_parameters={
                "min_silence_duration_ms": 300
            }
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        print("\r" + " " * 80, end="\r")

        if text:
            print("🇪🇸", text)

except KeyboardInterrupt:

    print("\nArrêt.")


finally:

    stream.stop_stream()
    stream.close()
    p.terminate()