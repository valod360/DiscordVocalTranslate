import io
import wave

import pyaudiowpatch as pyaudio

from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================
# CONFIGURATION
# =========================

DEVICE_INDEX = 23
RECORD_SECONDS = 4
CHUNK = 1024

WHISPER_MODEL = "turbo"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-es-fr"


# =========================
# WHISPER
# =========================

print("Chargement de Whisper...")

whisper = WhisperModel(
    WHISPER_MODEL,
    device="cuda",
    compute_type="float16"
)

print("Whisper chargé sur le GPU ✅")


# =========================
# TRADUCTION ES -> FR
# =========================

print("Chargement du traducteur ES → FR...")

tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATION_MODEL
)

translator = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL
)

translator.eval()

print("Traducteur chargé ✅")


def translate_es_fr(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    outputs = translator.generate(
        **inputs,
        max_new_tokens=256,
        num_beams=4
    )

    translated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return translated


# =========================
# AUDIO
# =========================

p = pyaudio.PyAudio()

device = p.get_device_info_by_index(
    DEVICE_INDEX
)

RATE = int(device["defaultSampleRate"])
CHANNELS = int(device["maxInputChannels"])

print()
print("Périphérique :", device["name"])
print("Fréquence :", RATE)
print("Canaux :", CHANNELS)
print()

print("🎧 Traduction Discord ES → FR active")
print("Ctrl+C pour arrêter.")
print()


stream = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK
)


# =========================
# BOUCLE
# =========================

try:

    while True:

        frames = []

        print("🎧 Écoute...", end="", flush=True)

        for _ in range(
            int(RATE / CHUNK * RECORD_SECONDS)
        ):

            data = stream.read(
                CHUNK,
                exception_on_overflow=False
            )

            frames.append(data)

        print(
            "\r🧠 Transcription...",
            end="",
            flush=True
        )

        wav_buffer = io.BytesIO()

        with wave.open(
            wav_buffer,
            "wb"
        ) as wf:

            wf.setnchannels(CHANNELS)

            wf.setsampwidth(
                p.get_sample_size(
                    pyaudio.paInt16
                )
            )

            wf.setframerate(RATE)

            wf.writeframes(
                b"".join(frames)
            )

        wav_buffer.seek(0)

        segments, info = whisper.transcribe(

            wav_buffer,

            language="es",

            beam_size=1,

            vad_filter=True,

            vad_parameters={
                "min_silence_duration_ms": 300
            }
        )

        spanish = " ".join(
            segment.text.strip()
            for segment in segments
        )

        print(
            "\r" + " " * 100,
            end="\r"
        )

        if spanish:

            french = translate_es_fr(
                spanish
            )

            print(
                f"🇪🇸 {spanish}"
            )

            print(
                f"🇫🇷 {french}"
            )

            print()


except KeyboardInterrupt:

    print("\nArrêt.")


finally:

    stream.stop_stream()
    stream.close()
    p.terminate()