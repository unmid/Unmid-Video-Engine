---
title: TTV
emoji: 🏢
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
---

# 🎬 Unmid Video Engine

**Turn a single sentence into a finished, narrated, captioned video.**

Unmid Video Engine is an open-source text-to-video pipeline. You give it a topic; it writes a scene-by-scene script, narrates it, finds matching stock footage, burns in captions, mixes in music, and renders a ready-to-post MP4 — in 16:9 or 9:16.

No timeline editor, no manual b-roll hunting, no voice recording.

<!-- Replace with a real GIF/screenshot of a render -->
<!-- ![Demo](docs/demo.gif) -->

---

## How it works

```
Topic + tone
     │
     ▼
[1] Script generation ──► Groq (llama-3.3-70b-versatile)
     │                   Scene-tagged script: [Tag] + one narration line
     ▼
[2] Scene parsing ──────► Tag → stock search query, line → narration
     │
     ├──► [3] Media fetch ──► Pexels API (video preferred, image fallback)
     ├──► [4] Narration ────► Kokoro TTS (29 voices, adjustable tempo)
     │
     ▼
[5] Segment render ─────► Ken Burns / motion effect + wrapped captions
     │
     ▼
[6] Final assembly ─────► Concatenate, mix optional music, encode via FFmpeg
     │
     ▼
   video.mp4
```

Every step is plain Python — no external orchestration, no paid render service.

---

## Features

- **Script writing with tone control** — four narrator presets: Funny / Dramatic / Casual / Educational. Prompts are tuned to strip out "AI voice" clichés (*delve*, *tapestry*, *in conclusion*) and vary sentence rhythm.
- **Automatic scene segmentation** — each scene is a `[Search Tag]` plus one line of narration, so the visual always matches what's being said.
- **29 neural voices (Kokoro TTS)** — American and British, male and female, with voice-tempo scaling from 0.6× to 1.4×.
- **Real stock footage** — Pexels video search with automatic image fallback, plus a generated solid-colour frame if a query returns nothing. The render never dead-ends.
- **Burned-in captions** — auto-wrapped, centred, adjustable font size (20–90px).
- **Cinematic motion** — slow zoom/pan on stills so photos don't feel static.
- **Optional background music** — upload an MP3 and control the mix level (0.00–0.40) so narration stays on top.
- **Aspect ratios** — `Full (16:9)` for YouTube, `Shorts (9:16)` for Reels / TikTok / Shorts.
- **Tunable render** — video-vs-photo ratio, target FPS (15–60), and FFmpeg speed preset for a quality/speed tradeoff.

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/unmid/unmid-video-engine.git
cd unmid-video-engine
```

### 2. Install

```bash
pip install -r requirements.txt
```

FFmpeg is required by MoviePy and `pydub`:

```bash
# Debian / Ubuntu
sudo apt install ffmpeg imagemagick espeak-ng python3-opencv

# macOS
brew install ffmpeg imagemagick espeak-ng
```

### 3. Get two free API keys

| Key | Where | Used for |
| --- | --- | --- |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Script generation |
| `PEXELS_API_KEY` | [pexels.com/api](https://www.pexels.com/api/) | Stock video & images |

```bash
export GROQ_API_KEY="gsk_..."
export PEXELS_API_KEY="..."
```

> The app starts and shows a banner if keys are missing, but rendering requires both.

### 4. Run

```bash
python app.py
```

Open the Gradio URL it prints, type a topic, hit **🎬 Render Cinematic Video Sequence**.

---

## Usage

1. **Concept** — e.g. *"The secret life of honeybees"*.
2. **Narrator tone** — Funny, Serious/Dramatic, Informal, or Educational.
3. **Dimensions** — 16:9 or 9:16.
4. **Captions** — on or off.
5. **Background music** *(optional)* — drop in an MP3.
6. **Advanced tuning** — voice, video/photo ratio, music volume, FPS, FFmpeg preset, voice tempo, caption size.
7. Render and download the MP4.

**Typical render time:** roughly one third of the final video length on CPU (a 60-second short usually finishes in 2–4 minutes), depending on FPS, preset, and how much stock footage has to be downloaded.

---

## Configuration

Tune behaviour by editing the constants at the top of `app.py`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Script-writing model |
| `TARGET_RESOLUTION` | `(1920, 1080)` | Output resolution |
| `video_clip_probability` | `0.65` | Chance a scene uses video over a still |
| `bg_music_volume` | `0.08` | Default music mix level |
| `fps` | `24` | Target frame rate |
| `font_size` | `45` | Default caption size |
| `VOICE_CHOICES` | 29 Kokoro voices | Voice picker mapping |

---

## Project structure

```
unmid-video-engine/
├── app.py            # Entire pipeline + Gradio interface
├── requirements.txt  # Python dependencies
└── packages.txt      # System packages (for Docker / Spaces builds)
```

The pipeline is deliberately single-file: `generate_script` → `parse_script` → `generate_media_asset` → `generate_tts_audio` → `compile_individual_segment` → `generate_video`. Each function is importable and reusable on its own.

---

## Deploy your own

The same repo runs as a Hugging Face Space. Add the front-matter block from [`README-huggingface.md`](README-huggingface.md) to a `README.md`, set `GROQ_API_KEY` and `PEXELS_API_KEY` as **Space secrets** (Settings → Variables and secrets), and make sure `packages.txt` ships `ffmpeg`, `imagemagick` and `espeak-ng`.

Free CPU Spaces sleep after inactivity — first request after a nap is slow while the TTS model loads.

---

## Limitations

- **Stock footage only.** There is no generative video step — visuals come from Pexels licensing, so check their terms for commercial use.
- **CPU rendering.** No GPU path yet; long videos are slow.
- **Template-driven pacing.** Scenes are cut to narration length, not beat-matched.
- **API dependency.** Script quality tracks the Groq model; missing or rate-limited keys stop a render.

---

## Roadmap

- [ ] Pluggable media providers (Pixabay, local library, generated images)
- [ ] Word-level caption timing instead of per-scene blocks
- [ ] Multiple output resolutions and a batch mode
- [ ] Optional GPU transcription/encoding path

---

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Unmid](https://github.com/unmid) · [Hugging Face](https://huggingface.co/Unmid)
