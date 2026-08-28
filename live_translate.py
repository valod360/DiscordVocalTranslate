import io
import wave
import queue
import threading
from collections import deque

import numpy as np
import torch
import pyaudiowpatch as pyaudio

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================================================
# CONFIGURATION
# =========================================================

DEVICE_INDEX = 23

WHISPER_MODEL = "turbo"
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

# Taille des petits blocs audio analysés
BLOCK_MS = 250

# Temps de silence avant de considérer une phrase terminée
END_SILENCE_MS = 700

# Audio conservé juste avant la détection de parole
# pour éviter de couper le début d'un mot
PRE_ROLL_MS = 500

# Durée maximale d'une phrase avant découpage automatique
MAX_PHRASE_MS = 12000


# =========================================================
# WHISPER
# =========================================================

print("Chargement de Whisper...")

whisper = WhisperModel(
    WHISPER_MODEL,
    device="cuda",
    compute_type="float16"
)

print("Whisper GPU ✅")


# =========================================================
# SILERO VAD
# =========================================================

print("Chargement de Silero VAD...")

vad_model = load_silero_vad()

print("Silero VAD ✅")


# =========================================================
# NLLB : ESPAGNOL -> FRANÇAIS
# =========================================================

print("Chargement du traducteur NLLB ES → FR...")

tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATION_MODEL,
    src_lang="spa_Latn"
)

translator = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL,
    dtype=torch.float16
).to("cuda")

translator.eval()

FRENCH_TOKEN_ID = tokenizer.convert_tokens_to_ids(
    "fra_Latn"
)

print("Traducteur NLLB GPU ✅")


# =========================================================
# AUDIO
# =========================================================

p = pyaudio.PyAudio()

device = p.get_device_info_by_index(
    DEVICE_INDEX
)

RATE = int(device["defaultSampleRate"])
CHANNELS = int(device["maxInputChannels"])

BLOCK_FRAMES = int(
    RATE * BLOCK_MS / 1000
)

print()
print("Périphérique :", device["name"])
print("Fréquence :", RATE)
print("Canaux :", CHANNELS)
print()


# =========================================================
# CONVERSION AUDIO POUR SILERO
# =========================================================

def audio_for_vad(data):

    audio = np.frombuffer(
        data,
        dtype=np.int16
    )

    # Stéréo -> mono
    if CHANNELS > 1:

        audio = audio.reshape(
            -1,
            CHANNELS
        )

        audio = audio.mean(
            axis=1
        )

    audio = audio.astype(
        np.float32
    )

    # Rééchantillonnage vers 16 kHz
    # Silero travaille en 16 kHz

    target_length = int(
        len(audio) * 16000 / RATE
    )

    if target_length <= 0:
        return torch.zeros(0)

    old_positions = np.arange(
        len(audio)
    )

    new_positions = np.linspace(
        0,
        len(audio) - 1,
        target_length
    )

    audio_16k = np.interp(
        new_positions,
        old_positions,
        audio
    )

    # int16 -> float [-1, 1]

    audio_16k /= 32768.0

    return torch.from_numpy(
        audio_16k.astype(np.float32)
    )


# =========================================================
# DÉTECTION DE PAROLE
# =========================================================

def contains_speech(data):

    wav = audio_for_vad(
        data
    )

    if len(wav) == 0:
        return False

    with torch.no_grad():

        timestamps = get_speech_timestamps(
            wav,
            vad_model,

            sampling_rate=16000,

            threshold=0.5,

            min_speech_duration_ms=100,

            min_silence_duration_ms=100
        )

    return len(timestamps) > 0


# =========================================================
# CRÉATION WAV EN MÉMOIRE POUR WHISPER
# =========================================================

def create_wav(pcm_data):

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb"
    ) as wf:

        wf.setnchannels(
            CHANNELS
        )

        # Int16 = 2 octets
        wf.setsampwidth(
            2
        )

        wf.setframerate(
            RATE
        )

        wf.writeframes(
            pcm_data
        )

    buffer.seek(0)

    return buffer


# =========================================================
# TRADUCTION ESPAGNOL -> FRANÇAIS
# =========================================================

def translate_es_fr(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to("cuda")

    with torch.inference_mode():

        outputs = translator.generate(
            **inputs,

            forced_bos_token_id=FRENCH_TOKEN_ID,

            max_length=256,

            num_beams=3
        )

    translated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return translated


# =========================================================
# FILE D'ATTENTE DES PHRASES
# =========================================================

phrase_queue = queue.Queue()

stop_event = threading.Event()


# =========================================================
# THREAD WHISPER + TRADUCTION
# =========================================================

def transcription_worker():

    while (
        not stop_event.is_set()
        or not phrase_queue.empty()
    ):

        try:

            pcm = phrase_queue.get(
                timeout=0.1
            )

        except queue.Empty:

            continue

        try:

            print()
            print("🧠 Transcription...")

            wav_buffer = create_wav(
                pcm
            )

            segments, info = whisper.transcribe(
                wav_buffer,

                language="es",

                beam_size=1,

                condition_on_previous_text=False,

                vad_filter=False
            )

            spanish = " ".join(
                segment.text.strip()
                for segment in segments
            ).strip()

            if not spanish:

                continue

            french = translate_es_fr(
                spanish
            )

            print()
            print("🇪🇸", spanish)
            print("🇫🇷", french)
            print()

        except Exception as e:

            print()
            print("❌ Erreur pendant la transcription/traduction :")
            print(e)
            print()

        finally:

            phrase_queue.task_done()


# =========================================================
# DÉMARRAGE DU THREAD
# =========================================================

worker = threading.Thread(
    target=transcription_worker,
    daemon=True
)

worker.start()


# =========================================================
# OUVERTURE DU LOOPBACK AUDIO
# =========================================================

stream = p.open(
    format=pyaudio.paInt16,

    channels=CHANNELS,

    rate=RATE,

    input=True,

    input_device_index=DEVICE_INDEX,

    frames_per_buffer=BLOCK_FRAMES
)


# =========================================================
# PRE-ROLL
# =========================================================

pre_roll_blocks = max(
    1,
    PRE_ROLL_MS // BLOCK_MS
)

pre_roll = deque(
    maxlen=pre_roll_blocks
)


# =========================================================
# ÉTAT DE LA CAPTURE
# =========================================================

recording = False

phrase = []

silence_ms = 0

phrase_duration_ms = 0


# =========================================================
# AFFICHAGE
# =========================================================

print("==============================================")
print("🎧 TRADUCTION DISCORD ESPAGNOL → FRANÇAIS")
print("==============================================")
print()
print("En attente d'une voix espagnole...")
print("Ctrl+C pour arrêter.")
print()


# =========================================================
# BOUCLE PRINCIPALE
# =========================================================

try:

    while True:

        # -------------------------------------------------
        # Lecture d'un petit bloc audio
        # -------------------------------------------------

        data = stream.read(
            BLOCK_FRAMES,
            exception_on_overflow=False
        )

        speech = contains_speech(
            data
        )


        # =================================================
        # PERSONNE NE PARLE ENCORE
        # =================================================

        if not recording:

            if speech:

                recording = True

                # On récupère le petit morceau précédant
                # la détection afin de ne pas couper
                # le début du premier mot.

                phrase = list(
                    pre_roll
                )

                phrase.append(
                    data
                )

                pre_roll.clear()

                silence_ms = 0

                phrase_duration_ms = BLOCK_MS

                print(
                    "🎙️ Parole détectée...      ",
                    end="\r",
                    flush=True
                )

            else:

                # On conserve continuellement les
                # dernières centaines de ms.

                pre_roll.append(
                    data
                )


        # =================================================
        # UNE PHRASE EST EN COURS
        # =================================================

        else:

            phrase.append(
                data
            )

            phrase_duration_ms += BLOCK_MS


            # -------------------------------------------------
            # Voix toujours présente
            # -------------------------------------------------

            if speech:

                silence_ms = 0


            # -------------------------------------------------
            # Silence
            # -------------------------------------------------

            else:

                silence_ms += BLOCK_MS


            # -------------------------------------------------
            # Fin naturelle d'une phrase
            # -------------------------------------------------

            finished = (
                silence_ms >= END_SILENCE_MS
            )


            # -------------------------------------------------
            # Phrase trop longue
            # -------------------------------------------------

            too_long = (
                phrase_duration_ms >= MAX_PHRASE_MS
            )


            # -------------------------------------------------
            # Envoi vers Whisper
            # -------------------------------------------------

            if finished or too_long:

                pcm = b"".join(
                    phrase
                )

                phrase_queue.put(
                    pcm
                )

                # Retour à l'état d'attente

                recording = False

                phrase = []

                silence_ms = 0

                phrase_duration_ms = 0

                pre_roll.clear()

                print(
                    "🎧 En attente...           ",
                    end="\r",
                    flush=True
                )


# =========================================================
# CTRL+C
# =========================================================

except KeyboardInterrupt:

    print()
    print()
    print("Arrêt demandé...")

    # Si quelqu'un était encore en train de parler,
    # on traite la dernière phrase avant de quitter.

    if phrase:

        phrase_queue.put(
            b"".join(phrase)
        )


# =========================================================
# NETTOYAGE
# =========================================================

finally:

    stream.stop_stream()
    stream.close()

    stop_event.set()

    # Attend que les dernières phrases soient traitées
    phrase_queue.join()

    p.terminate()

    print("Terminé.")