"""Clone a voice with SILMA TTS (Arabic + English) on a Modal GPU and save the speech as a WAV file.

    uv run modal run tts/silma_clone.py --text "Hello, how can I help you today?"           # default voice (texa)
    uv run modal run tts/silma_clone.py --voice vctk_p225 --text "Hello, how can I help you today?"
    uv run modal run tts/silma_clone.py --ref-audio voice.mp3 --text "Hello, how can I help you today?"
    uv run modal run tts/silma_clone.py --ref-audio voice.wav --ref-text "exact words in the clip" \
        --text-file reply.txt --out tts/output/reply.wav --nfe 8

Reference clip: 3-8 s of clean speech from one speaker, no music or background noise (any format ffmpeg reads).
Longer clips are cut to about 8 s. --ref-text is the exact transcript of the clip; leave it out and SILMA transcribes
the clip with Whisper (slower on first use, and a wrong transcript hurts quality).
--voice NAME picks a saved voice from tts/voices/ (default: texa) (NAME.wav/.flac/.mp3 plus NAME.txt with its transcript), e.g. texa,
hifitts_92 or vctk_p225. To save a new voice, drop its clip and transcript there under the same name.
Text can mix Arabic and English; Arabic without diacritics gets them added automatically.

Only clone voices you have the speaker's explicit permission to use (SILMA's license terms require it).
"""

import time
from pathlib import Path

import modal

GPU = "L40S"
REGION = "eu-west"  # same region as gemma4-serve-eu
OUT_DIR = Path(__file__).parent / "output"
VOICES_DIR = Path(__file__).parent / "voices"
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3")
DEFAULT_VOICE = "texa"


def resolve_voice(name: str) -> tuple[Path, str]:
    """Find a saved voice's clip and transcript in tts/voices/."""
    for suffix in AUDIO_SUFFIXES:
        clip = VOICES_DIR / f"{name}{suffix}"
        if clip.exists():
            transcript = clip.with_suffix(".txt")
            return clip, transcript.read_text(encoding="utf-8").strip() if transcript.exists() else ""
    names = sorted({p.stem for p in VOICES_DIR.iterdir() if p.suffix in AUDIO_SUFFIXES})
    raise SystemExit(f"No voice named {name!r} in {VOICES_DIR}. Available: {', '.join(names)}")


# Same image and cache as silma_bench.py, so neither is rebuilt or re-downloaded.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install("silma-tts==1.0.5")
    .env({"HF_HOME": "/cache/hf"})
)
cache_vol = modal.Volume.from_name("tts-cache", create_if_missing=True)

app = modal.App("silma-tts-clone")


@app.function(image=image, gpu=GPU, region=REGION, volumes={"/cache": cache_vol}, timeout=30 * 60)
def clone(ref_audio: bytes, ref_suffix: str, ref_text: str, text: str, nfe: int, speed: float, seed: int | None) -> dict:
    import io

    import soundfile as sf
    from silma_tts.api import SilmaTTS

    tts = SilmaTTS(hf_cache_dir="/cache/hf")
    ref_path = f"/tmp/ref{ref_suffix}"
    Path(ref_path).write_bytes(ref_audio)

    start = time.perf_counter()
    wav, sr, _ = tts.infer(ref_file=ref_path, ref_text=ref_text, gen_text=text, nfe_step=nfe, speed=speed,
                           seed=seed, progress=None)
    elapsed = time.perf_counter() - start
    cache_vol.commit()  # keeps the Whisper model if it was downloaded for transcription

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return {"wav": buf.getvalue(), "audio_s": len(wav) / sr, "gen_s": elapsed, "seed": tts.seed}


@app.local_entrypoint()
def main(voice: str = "", ref_audio: str = "", text: str = "", text_file: str = "", ref_text: str = "", out: str = "",
         nfe: int = 16, speed: float = 1.0, seed: int = -1):
    if bool(text) == bool(text_file):
        raise SystemExit("Pass exactly one of --text or --text-file")
    if voice and ref_audio:
        raise SystemExit("Pass only one of --voice or --ref-audio")
    if not ref_audio:
        ref, saved_text = resolve_voice(voice or DEFAULT_VOICE)
        ref_text = ref_text or saved_text
    else:
        ref = Path(ref_audio).expanduser()
    gen_text = text or Path(text_file).expanduser().read_text(encoding="utf-8")

    result = clone.remote(ref.read_bytes(), ref.suffix or ".wav", ref_text, gen_text, nfe, speed,
                          None if seed < 0 else seed)

    out_path = Path(out) if out else OUT_DIR / f"{ref.stem}_{int(time.time())}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(result["wav"])
    print(f"\nSaved {out_path} ({result['audio_s']:.1f} s of audio, generated in {result['gen_s']:.2f} s, "
          f"seed {result['seed']}; pass --seed {result['seed']} to reproduce)")
