from __future__ import annotations

import tempfile
from queue import Empty, Queue
from pathlib import Path
from threading import Event, Thread
from typing import Callable, Optional

from tar_lab.state import TARStateStore

try:
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    WhisperModel = None

try:
    import sounddevice as sd  # type: ignore
    import soundfile as sf  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sd = None  # type: ignore[assignment]
    sf = None  # type: ignore[assignment]


class SpeechProcessor:
    """Non-blocking speech processor for the Director control path."""

    def __init__(
        self,
        workspace: str = ".",
        wake_word: str = "lab",
        transcriber: Optional[Callable[[str], str]] = None,
        model_size: str = "small",
        device: str = "cpu",
    ):
        self.store = TARStateStore(workspace)
        self.wake_word = wake_word.lower()
        self._transcriber = transcriber
        self.model_size = model_size
        self.device = device
        self._queue: Queue[str] = Queue()
        self._commands: Queue[str] = Queue()
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._mic_thread: Optional[Thread] = None

    def _default_transcriber(self, audio_path: str) -> str:
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed")
        model = WhisperModel(self.model_size, device=self.device)
        segments, _ = model.transcribe(audio_path)
        return " ".join(segment.text for segment in segments).strip()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._mic_thread is not None:
            self._mic_thread.join(timeout=2.0)
            self._mic_thread = None

    def submit_audio(self, audio_path: str) -> None:
        self._queue.put(audio_path)

    def poll_command(self, timeout: float = 0.0) -> Optional[str]:
        try:
            if timeout > 0:
                return self._commands.get(timeout=timeout)
            return self._commands.get_nowait()
        except Empty:
            return None

    def capture_once(self, duration_s: float = 3.0, sample_rate: int = 16000) -> str:
        if sd is None or sf is None:
            raise RuntimeError("sounddevice and soundfile are required for live microphone capture")
        audio = sd.rec(int(duration_s * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        temp_dir = self.store.logs_dir / "voice"
        temp_dir.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=temp_dir)
        handle.close()
        sf.write(handle.name, audio, sample_rate)
        return handle.name

    def listen_once(self, duration_s: float = 3.0, sample_rate: int = 16000, timeout: float = 5.0) -> Optional[str]:
        self.start()
        audio_path = self.capture_once(duration_s=duration_s, sample_rate=sample_rate)
        self.submit_audio(audio_path)
        return self.poll_command(timeout=timeout)

    def start_microphone(self, duration_s: float = 3.0, sample_rate: int = 16000, pause_s: float = 0.25) -> None:
        if self._mic_thread is not None:
            return
        self.start()
        self._mic_thread = Thread(
            target=self._microphone_worker,
            kwargs={"duration_s": duration_s, "sample_rate": sample_rate, "pause_s": pause_s},
            daemon=True,
        )
        self._mic_thread.start()

    def _worker(self) -> None:
        transcriber = self._transcriber or self._default_transcriber
        while not self._stop.is_set():
            try:
                audio_path = self._queue.get(timeout=0.1)
            except Empty:
                continue
            text = transcriber(audio_path).strip()
            lowered = text.lower()
            self.store.append_audit_event("voice", "transcription", {"text": text})
            if self.wake_word in lowered:
                command = text[text.lower().find(self.wake_word) + len(self.wake_word) :].strip()
                if command:
                    self._commands.put(command)
                    self.store.append_audit_event("voice", "director_command", {"command": command})

    def _microphone_worker(self, duration_s: float, sample_rate: int, pause_s: float) -> None:
        while not self._stop.is_set():
            try:
                audio_path = self.capture_once(duration_s=duration_s, sample_rate=sample_rate)
                self.submit_audio(audio_path)
            except Exception as exc:
                self.store.append_audit_event("voice", "microphone_error", {"error": str(exc)})
                break
            self._stop.wait(pause_s)
