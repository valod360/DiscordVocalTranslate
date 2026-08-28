import pyaudiowpatch as pyaudio
import numpy as np

DEVICE_INDEX = 23
CHUNK = 1024

p = pyaudio.PyAudio()
device = p.get_device_info_by_index(DEVICE_INDEX)

print(f"Écoute de : {device['name']}")
print("Fais parler quelqu'un sur Discord. Ctrl+C pour arrêter.\n")

stream = p.open(
    format=pyaudio.paInt16,
    channels=int(device["maxInputChannels"]),
    rate=int(device["defaultSampleRate"]),
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK
)

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)

        audio = np.frombuffer(data, dtype=np.int16)

        volume = np.sqrt(np.mean(audio.astype(np.float32) ** 2))

        bars = int(min(volume / 300, 50))

        print(
            f"\rVolume : {volume:8.0f} | "
            + "█" * bars,
            end=""
        )

except KeyboardInterrupt:
    print("\nArrêt.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()