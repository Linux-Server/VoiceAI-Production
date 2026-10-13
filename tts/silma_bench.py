"""Benchmark SILMA TTS (150M, Arabic + English, voice cloning) on a Modal GPU in eu-west, next to the Gemma server.

Clones a reference voice, speaks English, Arabic and mixed sentences, and reports per-sentence latency and real-time
factor (RTF = generation time / audio duration). Also simulates sentence-by-sentence streaming of a multi-sentence
reply, the way a voice agent would feed Gemma's output to TTS, and reports time to first audio.
Generated audio is saved to tts/samples/ so quality can be judged by ear.

    uv run modal run tts/silma_bench.py                                   # default voice (tts/voices/texa)
    uv run modal run tts/silma_bench.py --ref-audio me.wav --ref-text "exact words spoken in me.wav"

The reference clip should be 3-8 s of clean speech (longer clips are cut to 8 s); ref-text must match it exactly.
SILMA's built-in streaming mode is broken in silma-tts 1.0.5, so streaming here is done per sentence.
"""

import re
import time
from pathlib import Path

import modal

GPU = "L40S"
REGION = "eu-west"  # same region as gemma4-serve-eu
NFE_STEPS = [16, 8]  # diffusion steps: 16 is SILMA's default; fewer is faster, possibly lower quality
SAMPLES_DIR = Path(__file__).parent / "samples"
DEFAULT_VOICE = Path(__file__).parent / "voices" / "texa.wav"  # its transcript is in texa.txt

# Bundled reference voice (Arabic, from the silma-tts package) and its exact transcript.
DEFAULT_REF_TEXT = "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."

SENTENCES = {
    "en_short": "Hi! How can I help you today?",
    "en_long": "Sure, I can book that for you. Your appointment is confirmed for Tuesday at three in the afternoon.",
    "ar_short": "مرحبا! كيف يمكنني مساعدتك اليوم؟",
    "ar_long": "بالتأكيد، يمكنني حجز ذلك لك. تم تأكيد موعدك يوم الثلاثاء الساعة الثالثة بعد الظهر.",
    "mixed": "Your order number is 4521, وسيصل الطلب خلال يومين.",
}
# A multi-sentence reply, streamed one sentence at a time.
STREAM_REPLY = {
    "en": "Great question. Dubai is warm in March, usually around twenty eight degrees. "
          "Evenings are cooler, so bring a light jacket. It's a great time to visit the beach.",
    "ar": "سؤال رائع. الطقس في دبي دافئ في شهر مارس، حوالي ثمان وعشرين درجة. "
          "تكون الأمسيات أبرد، لذلك أحضر سترة خفيفة. إنه وقت رائع لزيارة الشاطئ.",
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install("silma-tts==1.0.5")
    .env({"HF_HOME": "/cache/hf"})
)
cache_vol = modal.Volume.from_name("tts-cache", create_if_missing=True)  # model weights, vocoder, tashkeel model

app = modal.App("silma-tts-bench")


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation (Latin and Arabic), keeping the punctuation."""
    return [s.strip() for s in re.split(r"(?<=[.!?؟])\s+", text) if s.strip()]


@app.function(image=image, gpu=GPU, region=REGION, volumes={"/cache": cache_vol}, timeout=30 * 60)
def bench(ref_audio: bytes | None, ref_text: str | None) -> dict:
    import io
    from importlib.resources import files

    import numpy as np
    import soundfile as sf
    import torch
    from silma_tts.api import SilmaTTS

    t = time.perf_counter()
    tts = SilmaTTS(hf_cache_dir="/cache/hf")
    load_s = time.perf_counter() - t
    cache_vol.commit()

    if ref_audio is None:
        ref_path = str(files("silma_tts").joinpath("infer/ref_audio_samples/ar.ref.24k.wav"))
        ref_text = DEFAULT_REF_TEXT
    else:
        ref_path = "/tmp/ref.wav"
        Path(ref_path).write_bytes(ref_audio)

    def speak(text, nfe):
        start = time.perf_counter()
        wav, sr, _ = tts.infer(ref_file=ref_path, ref_text=ref_text, gen_text=text, nfe_step=nfe,
                               seed=0, show_info=lambda *_: None, progress=None)
        elapsed = time.perf_counter() - start
        return np.asarray(wav), sr, elapsed

    def to_wav(wav, sr):
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        return buf.getvalue()

    # Warm-up: first calls pay for CUDA kernels, the tashkeel model and NeMo normalizer grammars.
    t = time.perf_counter()
    for text in (SENTENCES["en_short"], SENTENCES["ar_short"]):
        speak(text, NFE_STEPS[0])
    warmup_s = time.perf_counter() - t

    rows, audio = [], {}
    for nfe in NFE_STEPS:
        for name, text in SENTENCES.items():
            times = []
            for _ in range(3):
                wav, sr, elapsed = speak(text, nfe)
                times.append(elapsed)
            dur = len(wav) / sr
            med = sorted(times)[1]
            rows.append({"nfe": nfe, "sentence": name, "chars": len(text), "audio_s": round(dur, 2),
                         "latency_ms": round(med * 1000), "rtf": round(med / dur, 3)})
            audio[f"nfe{nfe}_{name}.wav"] = to_wav(wav, sr)

    # Sentence-by-sentence streaming: first audio is ready after the first sentence; the rest must keep up with playback.
    streams = []
    for nfe in NFE_STEPS:
        for lang, reply in STREAM_REPLY.items():
            start, played_until, stalls, parts = time.perf_counter(), None, 0.0, []
            first_audio = None
            for sentence in split_sentences(reply):
                wav, sr, _ = speak(sentence, nfe)
                ready = time.perf_counter() - start
                if first_audio is None:
                    first_audio, played_until = ready, ready
                elif ready > played_until:  # the listener heard silence while this sentence was generated
                    stalls += ready - played_until
                    played_until = ready
                played_until += len(wav) / sr
                parts.append(wav)
            full = np.concatenate(parts)
            streams.append({"nfe": nfe, "lang": lang, "sentences": len(parts), "audio_s": round(len(full) / sr, 2),
                            "first_audio_ms": round(first_audio * 1000), "stall_ms": round(stalls * 1000)})
            audio[f"nfe{nfe}_stream_{lang}.wav"] = to_wav(full, sr)

    return {"gpu": torch.cuda.get_device_name(), "load_s": round(load_s, 1), "warmup_s": round(warmup_s, 1),
            "rows": rows, "streams": streams, "audio": audio}


@app.local_entrypoint()
def main(ref_audio: str = "", ref_text: str = ""):
    if ref_audio and not ref_text:
        raise SystemExit("--ref-text is required with --ref-audio (the exact words spoken in the clip)")
    if not ref_audio:
        ref_audio, ref_text = str(DEFAULT_VOICE), DEFAULT_VOICE.with_suffix(".txt").read_text(encoding="utf-8").strip()
    result = bench.remote(Path(ref_audio).read_bytes(), ref_text)

    print(f"\nGPU: {result['gpu']} | model load {result['load_s']} s | warm-up {result['warmup_s']} s\n")
    print(f"{'nfe':>4} {'sentence':10} {'chars':>5} {'audio':>7} {'latency':>9} {'RTF':>6}")
    for r in result["rows"]:
        print(f"{r['nfe']:>4} {r['sentence']:10} {r['chars']:>5} {r['audio_s']:>6}s {r['latency_ms']:>7}ms {r['rtf']:>6}")
    print(f"\n{'nfe':>4} {'stream':6} {'sentences':>9} {'audio':>7} {'first audio':>12} {'stalls':>8}")
    for s in result["streams"]:
        print(f"{s['nfe']:>4} {s['lang']:6} {s['sentences']:>9} {s['audio_s']:>6}s {s['first_audio_ms']:>10}ms "
              f"{s['stall_ms']:>6}ms")

    SAMPLES_DIR.mkdir(exist_ok=True)
    for name, data in result["audio"].items():
        (SAMPLES_DIR / name).write_bytes(data)
    print(f"\nSaved {len(result['audio'])} WAV files to {SAMPLES_DIR}")
