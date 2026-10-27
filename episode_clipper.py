from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parent
INPUT_DIR = BASE / "input"
OUTPUT_DIR = BASE / "output"
TEMP_DIR = BASE / "temp"
GAMEPLAY = BASE / "gameplay.mp4"
CONFIG_PATH = BASE / "config.json"


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Candidate:
    start: float
    end: float
    score: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def require(name: str) -> None:
    if not shutil.which(name):
        raise SystemExit(f"Не найден {name}. Проверьте установку и PATH.")


def load_config() -> dict:
    defaults = {
        "clip_count": 6,
        "min_duration": 28,
        "max_duration": 42,
        "model": "small",
        "language": "ru",
        "skip_start_seconds": 45,
        "skip_end_seconds": 30,
        "movie_height": 770,
        "movie_top": 80,
        "game_top": 858,
    }
    if CONFIG_PATH.exists():
        defaults.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return defaults


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def find_episode() -> Path:
    INPUT_DIR.mkdir(exist_ok=True)
    files = []
    for pattern in ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"):
        files.extend(INPUT_DIR.glob(pattern))
    if not files:
        raise SystemExit("Положите полную серию в папку input и снова запустите RUN.bat")
    files.sort(key=lambda p: p.name.lower())
    if len(files) > 1:
        print(f"В input найдено несколько файлов. Обрабатываю первый: {files[0].name}")
    return files[0]


def transcribe(episode: Path, model_name: str, language: str) -> list[Word]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit("Модуль faster-whisper не установлен.")

    print(f"\nЗагружаю Whisper {model_name}. При первом запуске модель будет скачана…")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(episode), language=language, word_timestamps=True,
        vad_filter=True, beam_size=5, condition_on_previous_text=True
    )
    words: list[Word] = []
    for segment in segments:
        print(f"Распознано до {segment.end/60:.1f} мин", end="\r", flush=True)
        for item in segment.words or []:
            text = (item.word or "").strip()
            if text and item.start is not None and item.end is not None:
                words.append(Word(float(item.start), float(item.end), text))
    print(f"\nВсего распознано слов: {len(words)}")
    return words


def punctuation_bonus(text: str) -> float:
    return text.count("?") * 7 + text.count("!") * 5 + text.count(":") * 1.5


def keyword_bonus(text: str) -> float:
    keys = (
        "почему", "зачем", "кто", "что ты", "серьезно", "правда", "никогда",
        "сейчас", "стой", "подожди", "нет", "давай", "уволен", "шеф",
        "дурак", "идиот", "люблю", "ненавижу", "проблем", "слушай"
    )
    low = text.lower()
    return sum(1.8 for key in keys if key in low)


def build_candidates(words: list[Word], total: float, cfg: dict) -> list[Candidate]:
    min_d = float(cfg["min_duration"])
    max_d = float(cfg["max_duration"])
    lo = float(cfg["skip_start_seconds"])
    hi = max(lo + max_d, total - float(cfg["skip_end_seconds"]))
    valid = [w for w in words if w.start >= lo and w.end <= hi]
    if len(valid) < 30:
        valid = words

    candidates: list[Candidate] = []
    # Start near a sentence boundary or after a short pause, sampled to avoid near-duplicates.
    start_indices = []
    last_start = -999.0
    for i, w in enumerate(valid):
        prev = valid[i - 1] if i else None
        boundary = i == 0 or (prev and (w.start - prev.end > 0.35 or prev.text.endswith((".", "!", "?"))))
        if boundary and w.start - last_start >= 4.0:
            start_indices.append(i)
            last_start = w.start

    for i in start_indices:
        start = max(0.0, valid[i].start - 0.15)
        target_end = start + max_d
        j = i
        while j + 1 < len(valid) and valid[j + 1].end <= target_end:
            j += 1
        while j > i and valid[j].end - start > max_d:
            j -= 1
        if j <= i:
            continue
        end = valid[j].end + 0.25
        # Prefer ending at punctuation while staying above minimum duration.
        k = j
        while k > i and valid[k].end - start >= min_d:
            if valid[k].text.endswith((".", "!", "?")):
                end = valid[k].end + 0.25
                j = k
                break
            k -= 1
        duration = end - start
        if not (min_d <= duration <= max_d + 0.5):
            continue
        chunk = valid[i:j + 1]
        if len(chunk) < 38:
            continue
        gaps = [chunk[x + 1].start - chunk[x].end for x in range(len(chunk) - 1)]
        longest_gap = max(gaps, default=0.0)
        if longest_gap > 3.0:
            continue
        text = " ".join(w.text for w in chunk)
        density = len(chunk) / duration
        score = density * 14 + punctuation_bonus(text) + keyword_bonus(text)
        score -= max(0.0, longest_gap - 1.0) * 8
        if chunk[0].text.lower() in {"и", "а", "но", "вот", "ну"}:
            score -= 2
        candidates.append(Candidate(start, min(end, total), score, text))
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def overlap_ratio(a: Candidate, b: Candidate) -> float:
    overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return overlap / min(a.duration, b.duration)


def select_candidates(candidates: list[Candidate], count: int) -> list[Candidate]:
    selected: list[Candidate] = []
    for candidate in candidates:
        if all(overlap_ratio(candidate, old) < 0.15 and abs(candidate.start - old.start) > 25 for old in selected):
            selected.append(candidate)
        if len(selected) >= count:
            break
    return sorted(selected, key=lambda c: c.start)


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def ass_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def write_ass(words: list[Word], clip: Candidate, path: Path) -> None:
    selected = [w for w in words if w.end > clip.start and w.start < clip.end]
    groups: list[list[Word]] = []
    current: list[Word] = []
    for w in selected:
        if current and (len(current) >= 6 or w.start - current[-1].end > 0.65):
            groups.append(current); current = []
        current.append(w)
        if w.text.endswith((".", "!", "?", ":", ";")) and len(current) >= 3:
            groups.append(current); current = []
    if current:
        groups.append(current)

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Trebuchet MS,82,&H00FFFFFF,&H000000FF,&H00000000,&H70000000,-1,0,0,0,100,100,0,0,1,7,2,2,52,52,1195,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for group in groups:
        start = max(0.0, group[0].start - clip.start)
        end = min(clip.duration, max(group[-1].end - clip.start, start + 0.35))
        text = " ".join(w.text for w in group)
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{ass_escape(text)}\n")
    path.write_text("".join(lines), encoding="utf-8-sig")


def ffmpeg_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def render_clip(episode: Path, words: list[Word], clip: Candidate, index: int, cfg: dict, gameplay_duration: float) -> Path:
    ass = TEMP_DIR / f"clip_{index:02d}.ass"
    out = OUTPUT_DIR / f"clip_{index:02d}_{int(clip.start//60):02d}m{int(clip.start%60):02d}s.mp4"
    write_ass(words, clip, ass)
    bg_max = max(0.0, gameplay_duration - clip.duration - 1)
    rng = random.Random(int(clip.start * 1000))
    bg_start = rng.uniform(0, bg_max) if bg_max else 0.0
    movie_height = int(cfg["movie_height"])
    movie_top = int(cfg["movie_top"])
    game_top = int(cfg["game_top"])
    game_height = 1920 - game_top
    ass_path = ffmpeg_filter_path(ass)
    vf = (
        "color=c=#05060A:s=1080x1920:r=30[canvas];"
        "[1:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        f"crop=1000:{game_height}:40:380,scale=1080:{game_height},"
        "eq=brightness=-0.03:saturation=0.92[game];"
        f"[canvas][game]overlay=0:{game_top}[withgame];"
        f"[0:v]scale=1080:{movie_height}:force_original_aspect_ratio=increase,"
        f"crop=1080:{movie_height},setsar=1[main];"
        f"[withgame][main]overlay=0:{movie_top}[base];"
        f"[base]drawbox=x=0:y={game_top-8}:w=1080:h=8:color=#05060A:t=fill,"
        f"ass='{ass_path}'[v]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{clip.start:.3f}", "-t", f"{clip.duration:.3f}", "-i", str(episode),
        "-stream_loop", "-1", "-ss", f"{bg_start:.3f}", "-t", f"{clip.duration:.3f}", "-i", str(GAMEPLAY),
        "-filter_complex", vf, "-map", "[v]", "-map", "0:a:0?",
        "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-t", f"{clip.duration:.3f}",
        "-movflags", "+faststart", str(out)
    ]
    print(f"\nРендер {index}: {clip.start:.1f}–{clip.end:.1f} сек → {out.name}")
    subprocess.run(cmd, check=True)
    return out


def main() -> None:
    require("ffmpeg"); require("ffprobe")
    if not GAMEPLAY.exists():
        raise SystemExit("Не найден gameplay.mp4 рядом с RUN.bat")
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    for old in OUTPUT_DIR.glob("clip_*.mp4"):
        old.unlink()

    cfg = load_config()
    episode = find_episode()
    total = probe_duration(episode)
    gameplay_duration = probe_duration(GAMEPLAY)
    print(f"Серия: {episode.name} ({total/60:.1f} мин)")
    print(f"Фон: gameplay.mp4 ({gameplay_duration/60:.1f} мин)")

    words = transcribe(episode, str(cfg["model"]), str(cfg["language"]))
    candidates = build_candidates(words, total, cfg)
    selected = select_candidates(candidates, int(cfg["clip_count"]))
    if not selected:
        raise SystemExit("Не удалось найти подходящие фрагменты. Попробуйте другую серию или уменьшите min_duration в config.json")
    print(f"Найдено кандидатов: {len(candidates)}. Выбрано: {len(selected)}")

    manifest = [f"Исходник: {episode.name}\n"]
    outputs = []
    for idx, clip in enumerate(selected, 1):
        outputs.append(render_clip(episode, words, clip, idx, cfg, gameplay_duration))
        manifest.append(
            f"\nКлип {idx:02d}: {clip.start:.1f}–{clip.end:.1f} сек, оценка {clip.score:.1f}\n{clip.text}\n"
        )
    (OUTPUT_DIR / "clips_manifest.txt").write_text("".join(manifest), encoding="utf-8")
    print("\nГОТОВО. Файлы находятся в папке output:")
    for path in outputs:
        print(" -", path.name)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nFFmpeg завершился с ошибкой, код {exc.returncode}")
        sys.exit(exc.returncode)
