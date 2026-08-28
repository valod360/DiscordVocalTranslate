import html
import io
import json
import os
import queue
import signal
import sys
import threading
import time
import wave

from pathlib import Path
from collections import deque

import numpy as np
import psutil
import torch

from proctap import ProcessAudioCapture

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from PySide6.QtCore import Qt, QObject, Signal, QTimer
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
# VALOD TRANSLATOR
# Discord-only process capture
# =========================================================


# =========================================================
# AUDIO
# ProcTap fournit du 48 kHz / stéréo / float32
# =========================================================

RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2  # après conversion float32 -> int16

BLOCK_MS = 250
BLOCK_FRAMES = int(RATE * BLOCK_MS / 1000)

BYTES_PER_BLOCK = (
    BLOCK_FRAMES
    * CHANNELS
    * SAMPLE_WIDTH
)


# =========================================================
# DÉTECTION DE PHRASES
# =========================================================

END_SILENCE_MS = 700
PRE_ROLL_MS = 500
MAX_PHRASE_MS = 12000


# =========================================================
# IA
# =========================================================

WHISPER_MODEL = "turbo"
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"


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

APPDATA = os.getenv(
    "APPDATA",
    str(Path.home())
)

APP_DIR = (
    Path(APPDATA)
    / "ValodTranslator"
)

APP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

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
# CONFIG
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

            saved = json.load(f)

        default.update(saved)

    except Exception:
        pass

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
            "Erreur config :",
            e
        )


# =========================================================
# LANGUE COURANTE
# =========================================================

direction_lock = threading.Lock()


def get_direction():

    with direction_lock:

        return (
            config["source"],
            config["target"]
        )


# =========================================================
# SIGNAUX QT
# =========================================================

class SubtitleSignals(QObject):

    translation_ready = Signal(
        str,
        str,
        str,
        str
    )

    status_changed = Signal(str)

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

    print(
        f"\n🔄 {source.upper()} → "
        f"{target.upper()}\n"
    )


# =========================================================
# OVERLAY
# =========================================================

class SubtitleOverlay(QWidget):

    def __init__(self):

        super().__init__()

        self.history = deque(
            maxlen=MAX_SUBTITLES
        )

        self.locked = config.get(
            "locked",
            True
        )

        self.drag_offset = None


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

        layout.setSpacing(6)


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

        mode_font.setBold(True)

        self.mode_label.setFont(
            mode_font
        )

        layout.addWidget(
            self.mode_label
        )


        # =================================================
        # SOUS TITRES
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

        subtitle_font.setBold(True)

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
            "made by valod"
        )

        self.credit_label.setAlignment(
            Qt.AlignCenter
        )

        credit_font = QFont(
            "Segoe UI",
            9
        )

        credit_font.setItalic(True)

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

        self.setLayout(layout)


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
            and config.get("y") is not None
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


        source, target = get_direction()

        self.change_direction(
            source,
            target
        )

        self.apply_lock_state()


    # =====================================================
    # STYLE
    # =====================================================

    def update_style(self):

        border = (
            "none"
            if self.locked
            else "2px solid rgba(80,180,255,230)"
        )

        self.subtitle_label.setStyleSheet(
            f"""
            QLabel {{
                color: white;
                background-color: rgba(0,0,0,185);
                border-radius: 18px;
                border: {border};
                padding: 20px 30px;
            }}
            """
        )

        self.mode_label.setStyleSheet(
            """
            QLabel {
                color: rgba(255,255,255,220);
                background-color: rgba(0,0,0,150);
                border-radius: 10px;
                padding: 5px 15px;
            }
            """
        )


    # =====================================================
    # LOCK
    # =====================================================

    def apply_lock_state(self):

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

        config["locked"] = locked

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

        if event.button() == Qt.LeftButton:

            self.drag_offset = (
                event.globalPosition().toPoint()
                - self.frameGeometry().topLeft()
            )


    def mouseMoveEvent(
        self,
        event
    ):

        if (
            self.locked
            or self.drag_offset is None
        ):
            return

        if event.buttons() & Qt.LeftButton:

            self.move(
                event.globalPosition().toPoint()
                - self.drag_offset
            )


    def mouseReleaseEvent(
        self,
        event
    ):

        if self.locked:
            return

        if event.button() == Qt.LeftButton:

            self.drag_offset = None

            config["x"] = self.x()
            config["y"] = self.y()

            save_config()


    # =====================================================
    # POSITION
    # =====================================================

    def move_to_bottom(self):

        screen = QApplication.primaryScreen()

        if not screen:
            return

        geometry = screen.availableGeometry()

        x = (
            geometry.x()
            + (
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

        self.move(x, y)

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

        src = LANGUAGES[source]
        dst = LANGUAGES[target]

        self.mode_label.setText(
            f"{src['flag']} {src['short']}"
            f"  →  "
            f"{dst['flag']} {dst['short']}"
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
    # TRADUCTION
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
            or target != current_target
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

            if i < len(history) - 1:

                lines.append(
                    '<span style="'
                    'font-size:17px;'
                    'color:#BBBBBB;">'
                    f'{sentence}'
                    '</span>'
                )

            else:

                lines.append(
                    '<span style="'
                    'font-size:24px;'
                    'color:white;">'
                    f'{sentence}'
                    '</span>'
                )

        self.subtitle_label.setText(
            "<br>".join(lines)
        )


# =========================================================
# GPU
# =========================================================

USE_CUDA = torch.cuda.is_available()

TORCH_DEVICE = (
    "cuda"
    if USE_CUDA
    else "cpu"
)

TORCH_DTYPE = (
    torch.float16
    if USE_CUDA
    else torch.float32
)

WHISPER_DEVICE = (
    "cuda"
    if USE_CUDA
    else "cpu"
)

WHISPER_COMPUTE = (
    "float16"
    if USE_CUDA
    else "int8"
)


print()
print("========================================")
print("VALOD TRANSLATOR")
print("made by valod")
print("========================================")
print()


if USE_CUDA:

    print(
        "GPU :",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "⚠️ Pas de CUDA, utilisation CPU."
    )


# =========================================================
# WHISPER
# =========================================================

print(
    "Chargement de Whisper..."
)

whisper = WhisperModel(
    WHISPER_MODEL,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE
)

print(
    "Whisper ✅"
)


# =========================================================
# SILERO
# =========================================================

print(
    "Chargement de Silero VAD..."
)

vad_model = load_silero_vad()

print(
    "Silero VAD ✅"
)


# =========================================================
# NLLB
# =========================================================

print(
    "Chargement de NLLB..."
)

translator = (
    AutoModelForSeq2SeqLM
    .from_pretrained(
        TRANSLATION_MODEL,
        dtype=TORCH_DTYPE
    )
    .to(TORCH_DEVICE)
)

translator.eval()


tokenizers = {}

for code, language in LANGUAGES.items():

    tokenizers[code] = (
        AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL,
            src_lang=language["nllb"]
        )
    )


print(
    "NLLB ✅"
)


# =========================================================
# TROUVER LE ROOT DISCORD
# =========================================================

def find_discord_root():

    candidates = []

    for proc in psutil.process_iter(
        ["pid", "ppid", "name"]
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

            if proc.ppid() not in ids:

                roots.append(
                    proc
                )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied
        ):

            pass


    if not roots:

        roots = candidates


    # Choisit le Discord racine possédant
    # le plus de descendants Discord.

    def score(proc):

        try:

            children = proc.children(
                recursive=True
            )

            return sum(
                1
                for child in children
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
# CONVERSION FLOAT32 -> INT16
# =========================================================

def float32_to_int16(
    pcm
):

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
        audio * 32767.0
    ).astype(
        np.int16
    )

    return audio.tobytes()


# =========================================================
# WAV WHISPER
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

    buffer.seek(0)

    return buffer


# =========================================================
# VAD
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
            0
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


    target_length = int(
        len(audio)
        * 16000
        / RATE
    )


    old_positions = np.arange(
        len(audio)
    )

    new_positions = np.linspace(
        0,
        len(audio) - 1,
        target_length
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


def contains_speech(
    data
):

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
# NLLB
# =========================================================

def translate_text(
    text,
    source,
    target
):

    tokenizer = tokenizers[
        source
    ]


    inputs = tokenizer(
        text,

        return_tensors="pt",

        truncation=True,

        max_length=512
    )


    inputs = {

        key: value.to(
            TORCH_DEVICE
        )

        for key, value
        in inputs.items()
    }


    target_token = (
        tokenizer.convert_tokens_to_ids(
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

stop_event = threading.Event()
reset_audio_event = threading.Event()


# =========================================================
# CALLBACK PROCTAP
# =========================================================

def discord_audio_callback(
    pcm,
    frames
):

    if stop_event.is_set():
        return

    if not pcm:
        return


    # ProcessAudioCapture fournit du float32
    # 48 kHz stéréo.

    int16_pcm = float32_to_int16(
        pcm
    )


    try:

        raw_audio_queue.put_nowait(
            int16_pcm
        )

    except queue.Full:

        pass


# =========================================================
# CAPTURE DISCORD
# =========================================================

def capture_worker():

    capture = None

    while not stop_event.is_set():

        discord = find_discord_root()


        if discord is None:

            signals.status_changed.emit(
                "Discord n'est pas lancé."
            )

            time.sleep(1)

            continue


        try:

            pid = discord.pid

            print()
            print(
                "🎧 Discord détecté"
            )

            print(
                "PID racine :",
                pid
            )

            print(
                "Capture : Discord + processus enfants"
            )

            print()


            signals.status_changed.emit(
                "Discord connecté • "
                "en attente d'une voix..."
            )


            reset_audio_event.set()


            capture = ProcessAudioCapture(
                pid=pid,
                on_data=discord_audio_callback,
                resample_quality="fast"
            )


            capture.start()


            while not stop_event.is_set():

                if not psutil.pid_exists(
                    pid
                ):

                    print(
                        "Discord fermé."
                    )

                    break


                if not capture.is_running:

                    print(
                        "Capture audio arrêtée."
                    )

                    break


                time.sleep(
                    0.5
                )


        except Exception as e:

            print()
            print(
                "❌ Erreur capture Discord :"
            )

            print(e)

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

            time.sleep(1)


# =========================================================
# VAD
# =========================================================

def vad_worker():

    pcm_buffer = bytearray()

    pre_roll_blocks = max(
        1,
        PRE_ROLL_MS // BLOCK_MS
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

        pcm_buffer = bytearray()

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

            chunk = raw_audio_queue.get(
                timeout=0.2
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
                    phrase_duration_ms = BLOCK_MS

                else:

                    pre_roll.append(
                        block
                    )


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


                if finished or too_long:

                    pcm = b"".join(
                        phrase
                    )


                    try:

                        phrase_queue.put_nowait(
                            pcm
                        )

                    except queue.Full:

                        pass


                    recording = False
                    phrase = []

                    silence_ms = 0
                    phrase_duration_ms = 0

                    pre_roll.clear()


# =========================================================
# WHISPER + NLLB
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

            source, target = (
                get_direction()
            )


            print(
                f"\n🧠 Transcription "
                f"{source.upper()}..."
            )


            wav_buffer = create_wav(
                pcm
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


            translated = translate_text(
                original,
                source,
                target
            )


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


            signals.translation_ready.emit(
                original,
                translated,
                source,
                target
            )


        except Exception as e:

            print(
                "❌ Erreur transcription :",
                e
            )


        finally:

            phrase_queue.task_done()


# =========================================================
# QT APP
# =========================================================

app = QApplication(
    sys.argv
)

app.setApplicationName(
    "Valod Translator"
)

app.setQuitOnLastWindowClosed(
    False
)


overlay = SubtitleOverlay()
overlay.show()


# =========================================================
# ICONE
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

icon_font.setBold(True)

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
# TRAY
# =========================================================

tray = QSystemTrayIcon(
    tray_icon,
    app
)

tray.setToolTip(
    "Valod Translator"
)

menu = QMenu()


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
# MODES
# =========================================================

translation_menu = menu.addMenu(
    "Sens de traduction"
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


for label, source, target in MODES:

    action = QAction(
        label,
        translation_menu
    )

    action.setCheckable(
        True
    )

    if (
        source == current_source
        and target == current_target
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
# CREDIT
# =========================================================

menu.addSeparator()

credit_action = QAction(
    "made by valod",
    menu
)

credit_action.setEnabled(
    False
)

menu.addAction(
    credit_action
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

    print(
        "\nArrêt..."
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
    daemon=True
)

vad_thread = threading.Thread(
    target=vad_worker,
    daemon=True
)

transcription_thread = threading.Thread(
    target=transcription_worker,
    daemon=True
)


capture_thread.start()
vad_thread.start()
transcription_thread.start()


print()
print(
    "🎧 Capture : Discord uniquement"
)
print(
    "YouTube / Opera / jeux ne doivent plus être capturés."
)
print()


exit_code = app.exec()


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