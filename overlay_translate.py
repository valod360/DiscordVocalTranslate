# =========================================================
# VALOD TRANSLATOR
# made by valod
# =========================================================

# =========================================================
# IMPORTANT :
# Les variables Hugging Face doivent être définies AVANT
# d'importer transformers / faster_whisper / huggingface_hub.
# =========================================================

import os
from pathlib import Path


LOCAL_APPDATA = Path(
    os.getenv(
        "LOCALAPPDATA",
        str(Path.home())
    )
)

APP_DIR = (
    LOCAL_APPDATA
    / "ValodTranslator"
)

APP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CACHE HUGGING FACE WINDOWS
# =========================================================

HF_CACHE = (
    APP_DIR
    / "huggingface"
)

HF_CACHE.mkdir(
    parents=True,
    exist_ok=True
)

os.environ["HF_HOME"] = str(
    HF_CACHE
)

os.environ["HF_HUB_CACHE"] = str(
    HF_CACHE / "hub"
)

# Évite les symlinks Windows nécessitant certains privilèges
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Supprime le warning associé
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# =========================================================
# IMPORTS
# =========================================================

import html
import io
import json
import queue
import signal
import sys
import threading
import time
import wave

from collections import deque

import numpy as np
import psutil
import torch

from proctap import ProcessAudioCapture

from faster_whisper import WhisperModel

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from PySide6.QtCore import (
    Qt,
    QObject,
    Signal,
    QTimer,
)

from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPixmap,
)

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMenu,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)


# =========================================================
# VERSION
# =========================================================

APP_NAME = "Valod Translator"
APP_VERSION = "0.1.2"


# =========================================================
# AUDIO
# =========================================================

RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2  # int16 après conversion

BLOCK_MS = 250

BLOCK_FRAMES = int(
    RATE
    * BLOCK_MS
    / 1000
)

BYTES_PER_BLOCK = (
    BLOCK_FRAMES
    * CHANNELS
    * SAMPLE_WIDTH
)


# =========================================================
# DÉTECTION DES PHRASES
# =========================================================

END_SILENCE_MS = 700

PRE_ROLL_MS = 500

MAX_PHRASE_MS = 12000


# =========================================================
# MODÈLES
# =========================================================

WHISPER_MODEL = "turbo"

TRANSLATION_MODEL = (
    "facebook/"
    "nllb-200-distilled-600M"
)


# =========================================================
# OVERLAY
# =========================================================

MAX_SUBTITLES = 3

OVERLAY_WIDTH = 1100
OVERLAY_HEIGHT = 310

BOTTOM_MARGIN = 130


# =========================================================
# CONFIG
# =========================================================

CONFIG_FILE = (
    APP_DIR
    / "config.json"
)


# =========================================================
# DISCORD
# =========================================================

DISCORD_NAMES = {

    "discord.exe",

    "discordptb.exe",

    "discordcanary.exe",

    "discorddevelopment.exe",
}


# =========================================================
# LANGUES
# =========================================================

LANGUAGES = {

    "es": {

        "name": "Espagnol",

        "short": "ES",

        "whisper": "es",

        "nllb": "spa_Latn",

        "flag": "🇪🇸",
    },


    "fr": {

        "name": "Français",

        "short": "FR",

        "whisper": "fr",

        "nllb": "fra_Latn",

        "flag": "🇫🇷",
    },


    "en": {

        "name": "Anglais",

        "short": "EN",

        "whisper": "en",

        "nllb": "eng_Latn",

        "flag": "🇬🇧",
    },
}


# =========================================================
# CONFIGURATION
# =========================================================

def load_config():

    default = {

        "source": "es",

        "target": "fr",

        "locked": True,

        "x": None,

        "y": None,
    }

    if not CONFIG_FILE.exists():

        return default


    try:

        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            saved = json.load(
                f
            )

        default.update(
            saved
        )


    except Exception as e:

        print(
            "⚠️ Erreur config :",
            e
        )


    return default


config = load_config()


def save_config():

    try:

        with open(
            CONFIG_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(

                config,

                f,

                indent=4,

                ensure_ascii=False
            )


    except Exception as e:

        print(
            "⚠️ Erreur sauvegarde config :",
            e
        )


# =========================================================
# DIRECTION DE TRADUCTION
# =========================================================

direction_lock = (
    threading.Lock()
)


def get_direction():

    with direction_lock:

        return (

            config["source"],

            config["target"],
        )


# =========================================================
# SIGNAUX QT
# =========================================================

class SubtitleSignals(
    QObject
):

    translation_ready = Signal(
        str,
        str,
        str,
        str
    )

    status_changed = Signal(
        str
    )

    direction_changed = Signal(
        str,
        str
    )


signals = SubtitleSignals()


def set_direction(
    source,
    target
):

    with direction_lock:

        config["source"] = source

        config["target"] = target


    save_config()


    signals.direction_changed.emit(
        source,
        target
    )


    print()

    print(
        "🔄 Traduction :",
        source.upper(),
        "→",
        target.upper()
    )

    print()


# =========================================================
# OVERLAY
# =========================================================

class SubtitleOverlay(
    QWidget
):

    def __init__(
        self
    ):

        super().__init__()


        self.history = deque(
            maxlen=MAX_SUBTITLES
        )


        self.locked = config.get(
            "locked",
            True
        )


        self.drag_offset = None


        # =================================================
        # FENÊTRE
        # =================================================

        self.setWindowFlags(

            Qt.FramelessWindowHint

            | Qt.WindowStaysOnTopHint

            | Qt.Tool
        )


        self.setAttribute(
            Qt.WA_TranslucentBackground
        )


        self.setAttribute(
            Qt.WA_ShowWithoutActivating
        )


        self.resize(
            OVERLAY_WIDTH,
            OVERLAY_HEIGHT
        )


        # =================================================
        # LAYOUT
        # =================================================

        layout = QVBoxLayout()


        layout.setContentsMargins(
            35,
            20,
            35,
            14
        )


        layout.setSpacing(
            6
        )


        # =================================================
        # MODE
        # =================================================

        self.mode_label = QLabel()


        self.mode_label.setAlignment(
            Qt.AlignCenter
        )


        mode_font = QFont(
            "Segoe UI",
            11
        )


        mode_font.setBold(
            True
        )


        self.mode_label.setFont(
            mode_font
        )


        layout.addWidget(
            self.mode_label
        )


        # =================================================
        # SOUS-TITRES
        # =================================================

        self.subtitle_label = QLabel(
            "Chargement..."
        )


        self.subtitle_label.setWordWrap(
            True
        )


        self.subtitle_label.setAlignment(
            Qt.AlignCenter
        )


        subtitle_font = QFont(
            "Segoe UI",
            22
        )


        subtitle_font.setBold(
            True
        )


        self.subtitle_label.setFont(
            subtitle_font
        )


        layout.addWidget(
            self.subtitle_label
        )


        # =================================================
        # CREDIT
        # =================================================

        self.credit_label = QLabel(
            f"made by valod • v{APP_VERSION}"
        )


        self.credit_label.setAlignment(
            Qt.AlignCenter
        )


        credit_font = QFont(
            "Segoe UI",
            9
        )


        credit_font.setItalic(
            True
        )


        self.credit_label.setFont(
            credit_font
        )


        self.credit_label.setStyleSheet(
            """
            QLabel {
                color: rgba(255,255,255,115);
                background: transparent;
                padding: 2px;
            }
            """
        )


        layout.addWidget(
            self.credit_label
        )


        self.setLayout(
            layout
        )


        # =================================================
        # SIGNALS
        # =================================================

        signals.translation_ready.connect(
            self.add_translation
        )


        signals.status_changed.connect(
            self.set_status
        )


        signals.direction_changed.connect(
            self.change_direction
        )


        # =================================================
        # POSITION
        # =================================================

        if (

            config.get("x") is not None

            and

            config.get("y") is not None
        ):

            self.move(

                config["x"],

                config["y"]
            )


        else:

            QTimer.singleShot(

                100,

                self.move_to_bottom
            )


        source, target = (
            get_direction()
        )


        self.change_direction(
            source,
            target
        )


        self.apply_lock_state()


    # =====================================================
    # STYLE
    # =====================================================

    def update_style(
        self
    ):

        if self.locked:

            border = "none"

        else:

            border = (
                "2px solid "
                "rgba(80,180,255,230)"
            )


        self.subtitle_label.setStyleSheet(
            f"""
            QLabel {{
                color: white;

                background-color:
                    rgba(0,0,0,185);

                border-radius: 18px;

                border: {border};

                padding: 20px 30px;
            }}
            """
        )


        self.mode_label.setStyleSheet(
            """
            QLabel {
                color:
                    rgba(255,255,255,220);

                background-color:
                    rgba(0,0,0,150);

                border-radius: 10px;

                padding: 5px 15px;
            }
            """
        )


    # =====================================================
    # LOCK
    # =====================================================

    def apply_lock_state(
        self
    ):

        self.setWindowFlag(

            Qt.WindowTransparentForInput,

            self.locked
        )


        self.update_style()


        self.show()


    def set_locked(
        self,
        locked
    ):

        self.locked = locked


        config["locked"] = (
            locked
        )


        save_config()


        self.apply_lock_state()


    # =====================================================
    # DRAG
    # =====================================================

    def mousePressEvent(
        self,
        event
    ):

        if self.locked:

            return


        if (
            event.button()
            == Qt.LeftButton
        ):

            self.drag_offset = (

                event
                .globalPosition()
                .toPoint()

                -

                self
                .frameGeometry()
                .topLeft()
            )


            event.accept()


    def mouseMoveEvent(
        self,
        event
    ):

        if (

            self.locked

            or

            self.drag_offset is None
        ):

            return


        if (
            event.buttons()
            & Qt.LeftButton
        ):

            self.move(

                event
                .globalPosition()
                .toPoint()

                -

                self.drag_offset
            )


            event.accept()


    def mouseReleaseEvent(
        self,
        event
    ):

        if self.locked:

            return


        if (
            event.button()
            == Qt.LeftButton
        ):

            self.drag_offset = None


            config["x"] = (
                self.x()
            )

            config["y"] = (
                self.y()
            )


            save_config()


            event.accept()


    # =====================================================
    # RECENTRER
    # =====================================================

    def move_to_bottom(
        self
    ):

        screen = (
            QApplication.primaryScreen()
        )


        if not screen:

            return


        geometry = (
            screen.availableGeometry()
        )


        x = (

            geometry.x()

            +

            (
                geometry.width()
                - self.width()
            ) // 2
        )


        y = (

            geometry.y()

            + geometry.height()

            - self.height()

            - BOTTOM_MARGIN
        )


        self.move(
            x,
            y
        )


        config["x"] = x

        config["y"] = y


        save_config()


    # =====================================================
    # DIRECTION
    # =====================================================

    def change_direction(
        self,
        source,
        target
    ):

        self.history.clear()


        src = LANGUAGES[
            source
        ]

        dst = LANGUAGES[
            target
        ]


        self.mode_label.setText(

            f"{src['flag']} "
            f"{src['short']}"

            "  →  "

            f"{dst['flag']} "
            f"{dst['short']}"
        )


        self.subtitle_label.setText(
            "En attente de Discord..."
        )


    # =====================================================
    # STATUS
    # =====================================================

    def set_status(
        self,
        text
    ):

        if not self.history:

            self.subtitle_label.setText(
                text
            )


    # =====================================================
    # TRADUCTION AFFICHÉE
    # =====================================================

    def add_translation(
        self,
        original,
        translated,
        source,
        target
    ):

        current_source, current_target = (
            get_direction()
        )


        if (

            source != current_source

            or

            target != current_target
        ):

            return


        self.history.append(
            translated
        )


        lines = []


        history = list(
            self.history
        )


        for i, sentence in enumerate(
            history
        ):

            sentence = html.escape(
                sentence
            )


            if (
                i
                < len(history) - 1
            ):

                lines.append(

                    '<span style="'

                    'font-size:17px;'

                    'color:#BBBBBB;'

                    '">'

                    f'{sentence}'

                    '</span>'
                )


            else:

                lines.append(

                    '<span style="'

                    'font-size:24px;'

                    'color:white;'

                    '">'

                    f'{sentence}'

                    '</span>'
                )


        self.subtitle_label.setText(

            "<br>".join(
                lines
            )
        )


# =========================================================
# HARDWARE
# =========================================================

CUDA_AVAILABLE = (
    torch.cuda.is_available()
)


GPU_NAME = None

GPU_MAJOR = 0
GPU_MINOR = 0

GPU_VRAM_GB = 0


if CUDA_AVAILABLE:

    try:

        GPU_NAME = (
            torch.cuda.get_device_name(
                0
            )
        )


        GPU_MAJOR, GPU_MINOR = (
            torch.cuda.get_device_capability(
                0
            )
        )


        GPU_VRAM_GB = (

            torch.cuda.get_device_properties(
                0
            ).total_memory

            / 1024**3
        )


    except Exception as e:

        print(
            "⚠️ Erreur détection GPU :",
            e
        )

        CUDA_AVAILABLE = False


# =========================================================
# INFO
# =========================================================

print()

print(
    "========================================"
)

print(
    "VALOD TRANSLATOR"
)

print(
    f"Version {APP_VERSION}"
)

print(
    "made by valod"
)

print(
    "========================================"
)

print()


if CUDA_AVAILABLE:

    print(
        "GPU détecté :",
        GPU_NAME
    )

    print(
        "Compute Capability :",
        f"{GPU_MAJOR}.{GPU_MINOR}"
    )

    print(
        "VRAM :",
        f"{GPU_VRAM_GB:.1f} Go"
    )

else:

    print(
        "GPU CUDA compatible non détecté."
    )


print()


# =========================================================
# WHISPER
# =========================================================

print(
    "Chargement de Whisper..."
)


whisper = None


# =========================================================
# GPU MODERNE
# =========================================================

if (
    CUDA_AVAILABLE
    and GPU_MAJOR >= 7
):

    try:

        print(
            "Mode Whisper choisi : "
            "CUDA float16"
        )


        whisper = WhisperModel(

            WHISPER_MODEL,

            device="cuda",

            compute_type="float16"
        )


        print(
            "Whisper CUDA float16 ✅"
        )


    except Exception as e:

        print(
            "⚠️ CUDA float16 impossible :",
            e
        )

        whisper = None


# =========================================================
# GPU PLUS ANCIEN
# GTX 1060 = Compute Capability 6.1
# =========================================================

if (
    whisper is None
    and CUDA_AVAILABLE
):

    try:

        print(
            "Mode Whisper choisi : "
            "CUDA int8"
        )


        whisper = WhisperModel(

            WHISPER_MODEL,

            device="cuda",

            compute_type="int8"
        )


        print(
            "Whisper CUDA int8 ✅"
        )


    except Exception as e:

        print(
            "⚠️ CUDA int8 impossible :",
            e
        )

        whisper = None


# =========================================================
# FALLBACK CPU
# =========================================================

if whisper is None:

    print(
        "Fallback Whisper : CPU int8"
    )


    whisper = WhisperModel(

        WHISPER_MODEL,

        device="cpu",

        compute_type="int8"
    )


    print(
        "Whisper CPU int8 ✅"
    )


# =========================================================
# SILERO VAD
# =========================================================

print(
    "Chargement de Silero VAD..."
)


vad_model = (
    load_silero_vad()
)


print(
    "Silero VAD ✅"
)


# =========================================================
# NLLB - CHOIX DU MATÉRIEL
# =========================================================

# Pour les GPU modernes avec suffisamment de VRAM,
# NLLB tourne sur le GPU.
#
# GTX 1060 et GPU anciens :
# NLLB sur CPU pour éviter FP16 lent / VRAM insuffisante.

TRANSLATOR_USE_CUDA = (

    CUDA_AVAILABLE

    and GPU_MAJOR >= 7

    and GPU_VRAM_GB >= 6
)


if TRANSLATOR_USE_CUDA:

    TRANSLATOR_DEVICE = "cuda"

    TRANSLATOR_DTYPE = (
        torch.float16
    )


else:

    TRANSLATOR_DEVICE = "cpu"

    TRANSLATOR_DTYPE = (
        torch.float32
    )


# =========================================================
# NLLB
# =========================================================

print(
    "Chargement de NLLB..."
)


if TRANSLATOR_USE_CUDA:

    print(
        "Mode NLLB : GPU float16"
    )

else:

    print(
        "Mode NLLB : CPU float32"
    )


translator = (

    AutoModelForSeq2SeqLM

    .from_pretrained(

        TRANSLATION_MODEL,

        dtype=TRANSLATOR_DTYPE
    )

    .to(
        TRANSLATOR_DEVICE
    )
)


translator.eval()


# =========================================================
# TOKENIZERS
# =========================================================

tokenizers = {}


for code, language in (
    LANGUAGES.items()
):

    tokenizers[code] = (

        AutoTokenizer

        .from_pretrained(

            TRANSLATION_MODEL,

            src_lang=
                language["nllb"]
        )
    )


print(
    "NLLB ✅"
)


# =========================================================
# TROUVER LE PROCESSUS RACINE DISCORD
# =========================================================

def find_discord_root():

    candidates = []


    for proc in psutil.process_iter(

        [
            "pid",
            "ppid",
            "name",
        ]
    ):

        try:

            name = (

                proc.info["name"]

                or ""
            ).lower()


            if name in DISCORD_NAMES:

                candidates.append(
                    proc
                )


        except (

            psutil.NoSuchProcess,

            psutil.AccessDenied
        ):

            pass


    if not candidates:

        return None


    ids = {

        proc.pid

        for proc in candidates
    }


    roots = []


    for proc in candidates:

        try:

            if (
                proc.ppid()
                not in ids
            ):

                roots.append(
                    proc
                )


        except Exception:

            pass


    if not roots:

        roots = candidates


    def score(
        proc
    ):

        try:

            children = (
                proc.children(
                    recursive=True
                )
            )


            return sum(

                1

                for child
                in children

                if (
                    child.name().lower()
                    in DISCORD_NAMES
                )
            )


        except Exception:

            return 0


    return max(
        roots,
        key=score
    )


# =========================================================
# CALLBACK PROCTAP
# =========================================================

def convert_proctap_audio(
    pcm,
    frames
):

    if not pcm:
        return None

    try:
        # ProcessAudioCapture renvoie toujours du PCM
        # 48 kHz / stéréo / float32.
        #
        # Le paramètre "frames" vaut actuellement -1
        # dans ProcTap, donc on l'ignore volontairement.

        audio = np.frombuffer(
            pcm,
            dtype=np.float32
        )

        if audio.size == 0:
            return None

        # Sécurité contre d'éventuelles valeurs
        # légèrement hors de [-1.0 ; +1.0]
        audio = np.clip(
            audio,
            -1.0,
            1.0
        )

        # float32 -> PCM int16
        audio_int16 = (
            audio * 32767.0
        ).astype(
            np.int16
        )

        return audio_int16.tobytes()

    except Exception as e:

        print(
            "⚠️ Erreur conversion audio ProcTap :",
            e
        )

        return None

        # =================================================
        # FLOAT32 STÉRÉO
        # 2 canaux × 4 octets = 8 octets/frame
        # =================================================

        if bytes_per_frame == 8:

            audio = np.frombuffer(
                pcm,
                dtype=np.float32
            )


            audio = np.clip(
                audio,
                -1.0,
                1.0
            )


            audio = (

                audio
                * 32767.0

            ).astype(
                np.int16
            )


            return audio.tobytes()


        # =================================================
        # INT16 STÉRÉO
        # =================================================

        elif bytes_per_frame == 4:

            return bytes(
                pcm
            )


        else:

            print(
                "⚠️ Format audio ProcTap inconnu :",
                bytes_per_frame,
                "octets/frame"
            )

            return None


    except Exception as e:

        print(
            "⚠️ Conversion audio :",
            e
        )

        return None


# =========================================================
# WAV POUR WHISPER
# =========================================================

def create_wav(
    pcm
):

    buffer = io.BytesIO()


    with wave.open(
        buffer,
        "wb"
    ) as wf:

        wf.setnchannels(
            CHANNELS
        )

        wf.setsampwidth(
            SAMPLE_WIDTH
        )

        wf.setframerate(
            RATE
        )

        wf.writeframes(
            pcm
        )


    buffer.seek(
        0
    )


    return buffer


# =========================================================
# AUDIO POUR SILERO
# =========================================================

def audio_for_vad(
    data
):

    audio = np.frombuffer(
        data,
        dtype=np.int16
    )


    if len(audio) == 0:

        return torch.zeros(
            0,
            dtype=torch.float32
        )


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


    # 48 kHz -> 16 kHz

    target_length = int(

        len(audio)

        * 16000

        / RATE
    )


    if target_length <= 0:

        return torch.zeros(
            0,
            dtype=torch.float32
        )


    old_positions = (
        np.arange(
            len(audio)
        )
    )


    new_positions = (
        np.linspace(

            0,

            len(audio) - 1,

            target_length
        )
    )


    audio = np.interp(

        new_positions,

        old_positions,

        audio
    )


    audio /= 32768.0


    return torch.from_numpy(

        audio.astype(
            np.float32
        )
    )


# =========================================================
# DÉTECTION DE PAROLE
# =========================================================

def contains_speech(
    data
):

    wav = audio_for_vad(
        data
    )


    if len(wav) == 0:

        return False


    with torch.no_grad():

        timestamps = (

            get_speech_timestamps(

                wav,

                vad_model,

                sampling_rate=16000,

                threshold=0.5,

                min_speech_duration_ms=100,

                min_silence_duration_ms=100
            )
        )


    return (
        len(timestamps)
        > 0
    )


# =========================================================
# TRADUCTION NLLB
# =========================================================

def translate_text(
    text,
    source,
    target
):

    tokenizer = (
        tokenizers[source]
    )


    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        max_length=512
    )


    inputs = {

        key:
            value.to(
                TRANSLATOR_DEVICE
            )

        for key, value
        in inputs.items()
    }


    target_token = (

        tokenizer

        .convert_tokens_to_ids(

            LANGUAGES[
                target
            ]["nllb"]
        )
    )


    with torch.inference_mode():

        outputs = translator.generate(

            **inputs,

            forced_bos_token_id=
                target_token,

            max_length=256,

            num_beams=3
        )


    return tokenizer.decode(

        outputs[0],

        skip_special_tokens=True
    )


# =========================================================
# QUEUES
# =========================================================

raw_audio_queue = queue.Queue(
    maxsize=300
)

phrase_queue = queue.Queue(
    maxsize=50
)


stop_event = (
    threading.Event()
)

reset_audio_event = (
    threading.Event()
)


# =========================================================
# CALLBACK DISCORD
# =========================================================

def discord_audio_callback(
    pcm,
    frames
):

    if stop_event.is_set():

        return


    converted = (
        convert_proctap_audio(
            pcm,
            frames
        )
    )


    if not converted:

        return


    try:

        raw_audio_queue.put_nowait(
            converted
        )


    except queue.Full:

        pass


# =========================================================
# CAPTURE DISCORD
# =========================================================

def capture_worker():

    capture = None


    while not stop_event.is_set():

        discord = (
            find_discord_root()
        )


        # =================================================
        # DISCORD PAS LANCÉ
        # =================================================

        if discord is None:

            signals.status_changed.emit(
                "Discord n'est pas lancé."
            )


            time.sleep(
                1
            )


            continue


        try:

            pid = (
                discord.pid
            )


            print()

            print(
                "🎧 Discord détecté"
            )

            print(
                "PID racine :",
                pid
            )

            print(
                "Capture : Discord uniquement"
            )

            print()


            signals.status_changed.emit(

                "Discord connecté • "
                "en attente d'une voix..."
            )


            reset_audio_event.set()


            capture = (
                ProcessAudioCapture(

                    pid=pid,

                    on_data=
                        discord_audio_callback
                )
            )


            capture.start()


            # =================================================
            # SURVEILLANCE
            # =================================================

            while (
                not stop_event.is_set()
            ):

                if not psutil.pid_exists(
                    pid
                ):

                    print(
                        "Discord fermé."
                    )

                    break


                try:

                    running = getattr(
                        capture,
                        "is_running",
                        True
                    )


                    if callable(
                        running
                    ):

                        running = (
                            running()
                        )


                    if running is False:

                        print(
                            "Capture Discord arrêtée."
                        )

                        break


                except Exception:

                    pass


                time.sleep(
                    0.5
                )


        except Exception as e:

            print()

            print(
                "❌ Erreur capture Discord :"
            )

            print(
                e
            )

            print()


            signals.status_changed.emit(
                "Erreur de capture Discord."
            )


        finally:

            if capture is not None:

                try:

                    capture.close()

                except Exception:

                    pass


                capture = None


            reset_audio_event.set()


        if not stop_event.is_set():

            time.sleep(
                1
            )


# =========================================================
# VAD / DÉCOUPAGE EN PHRASES
# =========================================================

def vad_worker():

    pcm_buffer = bytearray()


    pre_roll_blocks = max(

        1,

        PRE_ROLL_MS
        // BLOCK_MS
    )


    pre_roll = deque(
        maxlen=pre_roll_blocks
    )


    recording = False

    phrase = []

    silence_ms = 0

    phrase_duration_ms = 0


    def reset():

        nonlocal pcm_buffer

        nonlocal pre_roll

        nonlocal recording

        nonlocal phrase

        nonlocal silence_ms

        nonlocal phrase_duration_ms


        pcm_buffer = (
            bytearray()
        )


        pre_roll = deque(
            maxlen=pre_roll_blocks
        )


        recording = False

        phrase = []

        silence_ms = 0

        phrase_duration_ms = 0


    while not stop_event.is_set():

        if reset_audio_event.is_set():

            reset_audio_event.clear()

            reset()


        try:

            chunk = (
                raw_audio_queue.get(
                    timeout=0.2
                )
            )


        except queue.Empty:

            continue


        pcm_buffer.extend(
            chunk
        )


        while (

            len(pcm_buffer)

            >= BYTES_PER_BLOCK
        ):

            block = bytes(

                pcm_buffer[
                    :BYTES_PER_BLOCK
                ]
            )


            del pcm_buffer[
                :BYTES_PER_BLOCK
            ]


            speech = contains_speech(
                block
            )


            # =================================================
            # EN ATTENTE
            # =================================================

            if not recording:

                if speech:

                    recording = True


                    phrase = list(
                        pre_roll
                    )


                    phrase.append(
                        block
                    )


                    pre_roll.clear()


                    silence_ms = 0


                    phrase_duration_ms = (
                        BLOCK_MS
                    )


                else:

                    pre_roll.append(
                        block
                    )


            # =================================================
            # PHRASE EN COURS
            # =================================================

            else:

                phrase.append(
                    block
                )


                phrase_duration_ms += (
                    BLOCK_MS
                )


                if speech:

                    silence_ms = 0


                else:

                    silence_ms += (
                        BLOCK_MS
                    )


                finished = (

                    silence_ms

                    >= END_SILENCE_MS
                )


                too_long = (

                    phrase_duration_ms

                    >= MAX_PHRASE_MS
                )


                if (
                    finished
                    or too_long
                ):

                    pcm = b"".join(
                        phrase
                    )


                    source, target = (
                        get_direction()
                    )


                    try:

                        phrase_queue.put_nowait(
                            (
                                pcm,
                                source,
                                target
                            )
                        )


                    except queue.Full:

                        print(
                            "⚠️ Queue transcription pleine."
                        )


                    recording = False

                    phrase = []

                    silence_ms = 0

                    phrase_duration_ms = 0

                    pre_roll.clear()


# =========================================================
# WHISPER + TRADUCTION
# =========================================================

def transcription_worker():

    while (

        not stop_event.is_set()

        or

        not phrase_queue.empty()
    ):

        try:

            (
                pcm,
                source,
                target
            ) = phrase_queue.get(
                timeout=0.1
            )


        except queue.Empty:

            continue


        try:

            print()

            print(
                f"🧠 Transcription "
                f"{source.upper()}..."
            )


            wav_buffer = (
                create_wav(
                    pcm
                )
            )


            segments, info = (
                whisper.transcribe(

                    wav_buffer,

                    language=
                        LANGUAGES[
                            source
                        ]["whisper"],

                    beam_size=1,

                    condition_on_previous_text=False,

                    vad_filter=False
                )
            )


            original = " ".join(

                segment.text.strip()

                for segment
                in segments

            ).strip()


            if not original:

                continue


            translated = (
                translate_text(

                    original,

                    source,

                    target
                )
            )


            print()

            print(
                LANGUAGES[
                    source
                ]["flag"],
                original
            )


            print(
                LANGUAGES[
                    target
                ]["flag"],
                translated
            )


            print()


            signals.translation_ready.emit(

                original,

                translated,

                source,

                target
            )


        except Exception as e:

            print()

            print(
                "❌ Erreur transcription/traduction :"
            )

            print(
                e
            )

            print()


        finally:

            phrase_queue.task_done()


# =========================================================
# QT
# =========================================================

app = QApplication(
    sys.argv
)


app.setApplicationName(
    APP_NAME
)


app.setQuitOnLastWindowClosed(
    False
)


# =========================================================
# OVERLAY
# =========================================================

overlay = (
    SubtitleOverlay()
)

overlay.show()


# =========================================================
# ICÔNE TRAY
# =========================================================

pixmap = QPixmap(
    64,
    64
)


pixmap.fill(
    Qt.transparent
)


painter = QPainter(
    pixmap
)


painter.setRenderHint(
    QPainter.Antialiasing
)


painter.setBrush(

    QColor(
        45,
        120,
        255
    )
)


painter.setPen(
    Qt.NoPen
)


painter.drawEllipse(
    4,
    4,
    56,
    56
)


painter.setPen(

    QColor(
        255,
        255,
        255
    )
)


icon_font = QFont(
    "Segoe UI",
    24
)


icon_font.setBold(
    True
)


painter.setFont(
    icon_font
)


painter.drawText(

    pixmap.rect(),

    Qt.AlignCenter,

    "V"
)


painter.end()


tray_icon = QIcon(
    pixmap
)


# =========================================================
# SYSTEM TRAY
# =========================================================

tray = QSystemTrayIcon(

    tray_icon,

    app
)


tray.setToolTip(

    f"{APP_NAME} "
    f"v{APP_VERSION}"
)


menu = QMenu()


# =========================================================
# CAPTURE
# =========================================================

capture_action = QAction(

    "🎧 Capture : Discord uniquement",

    menu
)


capture_action.setEnabled(
    False
)


menu.addAction(
    capture_action
)


menu.addSeparator()


# =========================================================
# DEPLACEMENT
# =========================================================

move_action = QAction(

    "Déverrouiller / déplacer l'overlay",

    menu
)


move_action.setCheckable(
    True
)


move_action.setChecked(
    not overlay.locked
)


def toggle_move_mode(
    checked
):

    overlay.set_locked(
        not checked
    )


move_action.toggled.connect(
    toggle_move_mode
)


menu.addAction(
    move_action
)


menu.addSeparator()


# =========================================================
# LANGUES
# =========================================================

translation_menu = (
    menu.addMenu(
        "Sens de traduction"
    )
)


direction_group = QActionGroup(
    translation_menu
)


direction_group.setExclusive(
    True
)


MODES = [

    (
        "🇪🇸 ES → 🇫🇷 FR",
        "es",
        "fr"
    ),

    (
        "🇫🇷 FR → 🇪🇸 ES",
        "fr",
        "es"
    ),

    (
        "🇬🇧 EN → 🇫🇷 FR",
        "en",
        "fr"
    ),

    (
        "🇫🇷 FR → 🇬🇧 EN",
        "fr",
        "en"
    ),
]


current_source, current_target = (
    get_direction()
)


def make_language_callback(
    source,
    target
):

    def callback(
        checked=False
    ):

        if checked:

            set_direction(
                source,
                target
            )


    return callback


for (
    label,
    source,
    target
) in MODES:

    action = QAction(
        label,
        translation_menu
    )


    action.setCheckable(
        True
    )


    if (

        source
        == current_source

        and

        target
        == current_target
    ):

        action.setChecked(
            True
        )


    action.triggered.connect(

        make_language_callback(
            source,
            target
        )
    )


    direction_group.addAction(
        action
    )


    translation_menu.addAction(
        action
    )


# =========================================================
# RECENTRER
# =========================================================

menu.addSeparator()


recenter_action = QAction(

    "Recentrer l'overlay",

    menu
)


recenter_action.triggered.connect(
    overlay.move_to_bottom
)


menu.addAction(
    recenter_action
)


# =========================================================
# VERSION / CREDIT
# =========================================================

menu.addSeparator()


version_action = QAction(

    f"{APP_NAME} v{APP_VERSION}",

    menu
)


version_action.setEnabled(
    False
)


menu.addAction(
    version_action
)


credit_action = QAction(

    "made by valod",

    menu
)


credit_action.setEnabled(
    False
)



# =========================================================
# QUITTER
# =========================================================

menu.addSeparator()


quit_action = QAction(

    "Quitter",

    menu
)


menu.addAction(
    quit_action
)


tray.setContextMenu(
    menu
)


tray.show()


# =========================================================
# STOP
# =========================================================

def stop_program(
    *args
):

    if stop_event.is_set():

        return


    print()

    print(
        "Arrêt de Valod Translator..."
    )


    stop_event.set()


    tray.hide()


    app.quit()


quit_action.triggered.connect(
    stop_program
)


signal.signal(
    signal.SIGINT,
    stop_program
)


# Permet à Ctrl+C d'être traité par Qt
timer = QTimer()


timer.timeout.connect(
    lambda: None
)


timer.start(
    200
)


# =========================================================
# THREADS
# =========================================================

capture_thread = threading.Thread(

    target=capture_worker,

    daemon=True,

    name="DiscordCapture"
)


vad_thread = threading.Thread(

    target=vad_worker,

    daemon=True,

    name="VAD"
)


transcription_thread = threading.Thread(

    target=transcription_worker,

    daemon=True,

    name="Transcription"
)


capture_thread.start()

vad_thread.start()

transcription_thread.start()


# =========================================================
# READY
# =========================================================

print()

print(
    "========================================"
)

print(
    "🎧 VALOD TRANSLATOR ACTIF"
)

print(
    "========================================"
)

print()

print(
    "Capture : Discord uniquement"
)

print(
    f"Version : {APP_VERSION}"
)

print(
    "made by valod"
)

print()


# =========================================================
# QT LOOP
# =========================================================

exit_code = (
    app.exec()
)


# =========================================================
# CLEANUP
# =========================================================

stop_event.set()


capture_thread.join(
    timeout=3
)


vad_thread.join(
    timeout=2
)


transcription_thread.join(
    timeout=3
)


sys.exit(
    exit_code
)