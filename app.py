from kokoro import KPipeline

import soundfile as sf
import torch

import soundfile as sf
import os
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip
from PIL import Image
import tempfile
import random
import cv2
import math
import os, requests, io, time, re, random
from moviepy.editor import (
VideoFileClip, concatenate_videoclips, AudioFileClip, ImageClip,
CompositeVideoClip, TextClip, CompositeAudioClip
)
import gradio as gr
import shutil
import os
import moviepy.video.fx.all as vfx
import moviepy.config as mpy_config
from pydub import AudioSegment
from pydub.generators import Sine

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from bs4 import BeautifulSoup
import base64
from urllib.parse import quote
import pysrt
from gtts import gTTS
import gradio as gr # Import Gradio

# Import Kokoro
from kokoro import KPipeline

# Import moviepy components
from moviepy.editor import (
    VideoFileClip, concatenate_videoclips, AudioFileClip, ImageClip,
    CompositeVideoClip, TextClip, CompositeAudioClip, concatenate_audioclips
)
import moviepy.video.fx.all as vfx
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment
from pydub.generators import Sine

# ---------------- Secret Management ---------------- #
# Get secrets from environment variables (set in Gradio Space settings)
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY', '')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'openai/gpt-oss-120b:free')

# Check if secrets are loaded
if not PEXELS_API_KEY or not OPENROUTER_API_KEY:
    print("WARNING: API keys not found in environment variables!")
    print("Please set PEXELS_API_KEY and OPENROUTER_API_KEY in your Gradio Space secrets.")

# ---------------- Global Configuration ---------------- #
OUTPUT_VIDEO_FILENAME = "final_video.mp4"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Initialize Kokoro TTS pipeline (using American English)
try:
    pipeline = KPipeline(lang_code='a')
except Exception as e:
    print(f"Warning: Kokoro initialization failed: {e}")
    pipeline = None

# Global variables for video generation
selected_voice = 'af_heart'  # Default voice
voice_speed = 0.9  # Default voice speed
font_size = 45  # Default font size
video_clip_probability = 0.25  # Default probability for video clips
bg_music_volume = 0.08  # Default background music volume
fps = 30  # Default FPS
preset = "veryfast"  # Default preset
TARGET_RESOLUTION = None
CAPTION_COLOR = None
TEMP_FOLDER = None

# ---------------- Helper Functions ---------------- #

def generate_script(user_input):
    """Generate documentary script with proper OpenRouter handling."""
    if not OPENROUTER_API_KEY:
        print("OpenRouter API key not configured")
        return None
        
    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'HTTP-Referer': 'https://huggingface.co',  # Use HuggingFace as referer
        'X-Title': 'AI Documentary Maker'
    }

    prompt = f"""You're a professional documentary narrator. Your job is to write a serious, natural, and informative video script based on one topic.

The script should sound like a real human voiceover from a TV show or documentary — clear, factual, and engaging, like something you'd hear on National Geographic or a news report.

Structure:
- Break the script into scenes using [Tags]. Each tag is a short title (1–2 words) that describes the visual or idea.
- Under each tag, write one sentence (max 12 words) that fits the tag and continues the topic.
- The full script should make sense as one connected narration — no randomness.
- Use natural, formal English. No slang, no fake AI language, and no robotic tone.
- Do not use humor, sarcasm, or casual language. This is a serious narration.
- No emotion-sound words like "aww," "eww," "whoa," etc.
- Do not use numbers like 1, 2, 3 — write them out as one, two, three.
- At the end, add a [Subscribe] tag with a formal or respectful reason to follow or subscribe.

Only output the script. No extra comments or text.

Example:

[Ocean]

The ocean covers over seventy percent of the Earth's surface.

[Currents]

Ocean currents distribute heat and regulate global climate patterns.

[Coral Reefs]

These ecosystems support over one million species of marine life.

[Pollution]

Plastic waste threatens marine biodiversity and food chains.

[Climate Impact]

Rising temperatures are causing coral bleaching and habitat loss.

[Subscribe]

Follow to explore more about the changing planet we live on.

Now here is the Topic/script: {user_input}
"""

    data = {
        'model': OPENROUTER_MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.4,
        'max_tokens': 5000
    }

    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                print("Unexpected response format:", response_data)
                return None
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"Request failed: {str(e)}")
        return None

def parse_script(script_text):
    """Parse the generated script into a list of elements."""
    sections = {}
    current_title = None
    current_text = ""

    try:
        for line in script_text.splitlines():
            line = line.strip()
            if line.startswith("[") and "]" in line:
                bracket_start = line.find("[")
                bracket_end = line.find("]", bracket_start)
                if bracket_start != -1 and bracket_end != -1:
                    if current_title is not None:
                        sections[current_title] = current_text.strip()
                    current_title = line[bracket_start+1:bracket_end]
                    current_text = line[bracket_end+1:].strip()
            elif current_title:
                current_text += " " + line

        if current_title:
            sections[current_title] = current_text.strip()

        elements = []
        for title, narration in sections.items():
            if not title or not narration:
                continue

            media_element = {"type": "media", "prompt": title, "effects": "fade-in"}
            words = narration.split()
            duration = max(3, len(words) * 0.5)
            tts_element = {"type": "tts", "text": narration, "voice": "en", "duration": duration}
            elements.append(media_element)
            elements.append(tts_element)

        return elements
    except Exception as e:
        print(f"Error parsing script: {e}")
        return []

def search_pexels_videos(query, pexels_api_key):
    """Search for a video on Pexels by query and return a random HD video."""
    if not pexels_api_key:
        print("Pexels API key not configured")
        return None
        
    headers = {'Authorization': pexels_api_key}
    base_url = "https://api.pexels.com/videos/search"
    num_pages = 3
    videos_per_page = 15

    max_retries = 3
    retry_delay = 1

    search_query = query
    all_videos = []

    for page in range(1, num_pages + 1):
        for attempt in range(max_retries):
            try:
                params = {"query": search_query, "per_page": videos_per_page, "page": page}
                response = requests.get(base_url, headers=headers, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    videos = data.get("videos", [])

                    if not videos:
                        print(f"No videos found on page {page}.")
                        break

                    for video in videos:
                        video_files = video.get("video_files", [])
                        for file in video_files:
                            if file.get("quality") == "hd":
                                all_videos.append(file.get("link"))
                                break

                    break

                elif response.status_code == 429:
                    print(f"Rate limit hit (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Error fetching videos: {response.status_code} {response.text}")
                    break

            except requests.exceptions.RequestException as e:
                print(f"Request exception: {e}")
                break

    if all_videos:
        random_video = random.choice(all_videos)
        print(f"Selected random video from {len(all_videos)} HD videos")
        return random_video
    else:
        print("No suitable videos found after searching all pages.")
        return None

def search_pexels_images(query, pexels_api_key):
    """Search for an image on Pexels by query."""
    if not pexels_api_key:
        print("Pexels API key not configured")
        return None
        
    headers = {'Authorization': pexels_api_key}
    url = "https://api.pexels.com/v1/search"
    params = {"query": query, "per_page": 5, "orientation": "landscape"}

    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                photos = data.get("photos", [])
                if photos:
                    photo = random.choice(photos[:min(5, len(photos))])
                    img_url = photo.get("src", {}).get("original")
                    return img_url
                else:
                    print(f"No images found for query: {query}")
                    return None

            elif response.status_code == 429:
                print(f"Rate limit hit (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Error fetching images: {response.status_code} {response.text}")
                break

        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            break

    print(f"No Pexels images found for query: {query} after all attempts")
    return None

def search_google_images(query):
    """Search for images on Google Images (for news-related queries)"""
    try:
        search_url = f"https://www.google.com/search?q={quote(query)}&tbm=isch"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        img_tags = soup.find_all("img")
        image_urls = []
        for img in img_tags:
            src = img.get("src", "")
            if src.startswith("http") and "gstatic" not in src:
                image_urls.append(src)

        if image_urls:
            return random.choice(image_urls[:5]) if len(image_urls) >= 5 else image_urls[0]
        else:
            print(f"No Google Images found for query: {query}")
            return None
    except Exception as e:
        print(f"Error in Google Images search: {e}")
        return None

def download_image(image_url, filename):
    """Download an image from a URL to a local file with enhanced error handling."""
    try:
        headers = {"User-Agent": USER_AGENT}
        print(f"Downloading image from: {image_url}")
        response = requests.get(image_url, headers=headers, stream=True, timeout=15)
        response.raise_for_status()

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Image downloaded successfully to: {filename}")

        try:
            img = Image.open(filename)
            img.verify()
            img = Image.open(filename)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img.save(filename)
            print(f"Image validated and processed: {filename}")
            return filename
        except Exception as e_validate:
            print(f"Downloaded file is not a valid image: {e_validate}")
            if os.path.exists(filename):
                os.remove(filename)
            return None

    except requests.exceptions.RequestException as e_download:
        print(f"Image download error: {e_download}")
        if os.path.exists(filename):
            os.remove(filename)
        return None
    except Exception as e_general:
        print(f"General error during image processing: {e_general}")
        if os.path.exists(filename):
            os.remove(filename)
        return None

def download_video(video_url, filename):
    """Download a video from a URL to a local file."""
    try:
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Video downloaded successfully to: {filename}")
        return filename
    except Exception as e:
        print(f"Video download error: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return None

def generate_media(prompt, user_image=None, current_index=0, total_segments=1):
    """Generate a visual asset by searching for videos or images."""
    safe_prompt = re.sub(r'[^\w\s-]', '', prompt).strip().replace(' ', '_')

    if "news" in prompt.lower():
        print(f"News-related query detected: {prompt}. Using Google Images...")
        image_file = os.path.join(TEMP_FOLDER, f"{safe_prompt}_news.jpg")
        image_url = search_google_images(prompt)
        if image_url:
            downloaded_image = download_image(image_url, image_file)
            if downloaded_image:
                print(f"News image saved to {downloaded_image}")
                return {"path": downloaded_image, "asset_type": "image"}

    if random.random() < video_clip_probability:
        video_file = os.path.join(TEMP_FOLDER, f"{safe_prompt}_video.mp4")
        video_url = search_pexels_videos(prompt, PEXELS_API_KEY)
        if video_url:
            downloaded_video = download_video(video_url, video_file)
            if downloaded_video:
                print(f"Video asset saved to {downloaded_video}")
                return {"path": downloaded_video, "asset_type": "video"}

    image_file = os.path.join(TEMP_FOLDER, f"{safe_prompt}.jpg")
    image_url = search_pexels_images(prompt, PEXELS_API_KEY)
    if image_url:
        downloaded_image = download_image(image_url, image_file)
        if downloaded_image:
            print(f"Image asset saved to {downloaded_image}")
            return {"path": downloaded_image, "asset_type": "image"}

    # Fallback to generic images
    fallback_terms = ["nature", "people", "landscape", "technology", "business"]
    for term in fallback_terms:
        print(f"Trying fallback image search with term: {term}")
        fallback_file = os.path.join(TEMP_FOLDER, f"fallback_{term}.jpg")
        fallback_url = search_pexels_images(term, PEXELS_API_KEY)
        if fallback_url:
            downloaded_fallback = download_image(fallback_url, fallback_file)
            if downloaded_fallback:
                print(f"Fallback image saved to {downloaded_fallback}")
                return {"path": downloaded_fallback, "asset_type": "image"}

    print(f"Failed to generate visual asset for prompt: {prompt}")
    return None

def generate_silent_audio(duration, sample_rate=24000):
    """Generate a silent WAV audio file lasting 'duration' seconds."""
    num_samples = int(duration * sample_rate)
    silence = np.zeros(num_samples, dtype=np.float32)
    silent_path = os.path.join(TEMP_FOLDER, f"silent_{int(time.time())}.wav")
    sf.write(silent_path, silence, sample_rate)
    print(f"Silent audio generated: {silent_path}")
    return silent_path

def generate_tts(text, voice):
    """Generate TTS audio using Kokoro, falling back to gTTS or silent audio if needed."""
    safe_text = re.sub(r'[^\w\s-]', '', text[:10]).strip().replace(' ', '_')
    file_path = os.path.join(TEMP_FOLDER, f"tts_{safe_text}.wav")

    if os.path.exists(file_path):
        print(f"Using cached TTS for text '{text[:10]}...'")
        return file_path

    # Try Kokoro if available
    if pipeline:
        try:
            kokoro_voice = selected_voice if voice == 'en' else voice
            generator = pipeline(text, voice=kokoro_voice, speed=voice_speed, split_pattern=r'\n+')
            audio_segments = []
            for i, (gs, ps, audio) in enumerate(generator):
                audio_segments.append(audio)
            full_audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
            sf.write(file_path, full_audio, 24000)
            print(f"TTS audio saved to {file_path} (Kokoro)")
            return file_path
        except Exception as e:
            print(f"Error with Kokoro TTS: {e}")

    # Fallback to gTTS
    try:
        print("Falling back to gTTS...")
        tts = gTTS(text=text, lang='en')
        mp3_path = os.path.join(TEMP_FOLDER, f"tts_{safe_text}.mp3")
        tts.save(mp3_path)
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(file_path, format="wav")
        os.remove(mp3_path)
        print(f"Fallback TTS saved to {file_path} (gTTS)")
        return file_path
    except Exception as fallback_error:
        print(f"Both TTS methods failed: {fallback_error}")
        return generate_silent_audio(duration=max(3, len(text.split()) * 0.5))

def apply_kenburns_effect(clip, target_resolution, effect_type=None):
    """Apply a smooth Ken Burns effect with a single movement pattern."""
    target_w, target_h = target_resolution
    clip_aspect = clip.w / clip.h
    target_aspect = target_w / target_h

    if clip_aspect > target_aspect:
        new_height = target_h
        new_width = int(new_height * clip_aspect)
    else:
        new_width = target_w
        new_height = int(new_width / clip_aspect)

    clip = clip.resize(newsize=(new_width, new_height))
    base_scale = 1.15
    new_width = int(new_width * base_scale)
    new_height = int(new_height * base_scale)
    clip = clip.resize(newsize=(new_width, new_height))

    max_offset_x = new_width - target_w
    max_offset_y = new_height - target_h

    available_effects = ["zoom-in", "zoom-out", "pan-left", "pan-right", "up-left"]
    if effect_type is None or effect_type == "random":
        effect_type = random.choice(available_effects)

    if effect_type == "zoom-in":
        start_zoom = 0.9
        end_zoom = 1.1
        start_center = (new_width / 2, new_height / 2)
        end_center = start_center
    elif effect_type == "zoom-out":
        start_zoom = 1.1
        end_zoom = 0.9
        start_center = (new_width / 2, new_height / 2)
        end_center = start_center
    elif effect_type == "pan-left":
        start_zoom = 1.0
        end_zoom = 1.0
        start_center = (max_offset_x + target_w / 2, (max_offset_y // 2) + target_h / 2)
        end_center = (target_w / 2, (max_offset_y // 2) + target_h / 2)
    elif effect_type == "pan-right":
        start_zoom = 1.0
        end_zoom = 1.0
        start_center = (target_w / 2, (max_offset_y // 2) + target_h / 2)
        end_center = (max_offset_x + target_w / 2, (max_offset_y // 2) + target_h / 2)
    elif effect_type == "up-left":
        start_zoom = 1.0
        end_zoom = 1.0
        start_center = (max_offset_x + target_w / 2, max_offset_y + target_h / 2)
        end_center = (target_w / 2, target_h / 2)
    else:
        raise ValueError(f"Unsupported effect_type: {effect_type}")

    def transform_frame(get_frame, t):
        frame = get_frame(t)
        ratio = t / clip.duration if clip.duration > 0 else 0
        ratio = 0.5 - 0.5 * math.cos(math.pi * ratio)
        current_zoom = start_zoom + (end_zoom - start_zoom) * ratio
        crop_w = int(target_w / current_zoom)
        crop_h = int(target_h / current_zoom)
        current_center_x = start_center[0] + (end_center[0] - start_center[0]) * ratio
        current_center_y = start_center[1] + (end_center[1] - start_center[1]) * ratio
        min_center_x = crop_w / 2
        max_center_x = new_width - crop_w / 2
        min_center_y = crop_h / 2
        max_center_y = new_height - crop_h / 2
        current_center_x = max(min_center_x, min(current_center_x, max_center_x))
        current_center_y = max(min_center_y, min(current_center_y, max_center_y))
        cropped_frame = cv2.getRectSubPix(frame, (crop_w, crop_h), (current_center_x, current_center_y))
        resized_frame = cv2.resize(cropped_frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return resized_frame

    return clip.fl(transform_frame)

def resize_to_fill(clip, target_resolution):
    """Resize and crop a clip to fill the target resolution while maintaining aspect ratio."""
    target_w, target_h = target_resolution
    clip_aspect = clip.w / clip.h
    target_aspect = target_w / target_h

    if clip_aspect > target_aspect:
        clip = clip.resize(height=target_h)
        crop_amount = (clip.w - target_w) / 2
        clip = clip.crop(x1=crop_amount, x2=clip.w - crop_amount, y1=0, y2=clip.h)
    else:
        clip = clip.resize(width=target_w)
        crop_amount = (clip.h - target_h) / 2
        clip = clip.crop(x1=0, x2=clip.w, y1=crop_amount, y2=clip.h - crop_amount)

    return clip

def add_background_music(final_video, bg_music_volume=0.10):
    """Add background music to the final video using any MP3 file found."""
    try:
        bg_music_path = "music.mp3"
        if bg_music_path and os.path.exists(bg_music_path):
            print(f"Adding background music from: {bg_music_path}")
            bg_music = AudioFileClip(bg_music_path)
            if bg_music.duration < final_video.duration:
                loops_needed = math.ceil(final_video.duration / bg_music.duration)
                bg_segments = [bg_music] * loops_needed
                bg_music = concatenate_audioclips(bg_segments)
            bg_music = bg_music.subclip(0, final_video.duration)
            bg_music = bg_music.volumex(bg_music_volume)
            video_audio = final_video.audio
            mixed_audio = CompositeAudioClip([video_audio, bg_music])
            final_video = final_video.set_audio(mixed_audio)
            print("Background music added successfully")
        else:
            print("No MP3 files found, skipping background music")
        return final_video
    except Exception as e:
        print(f"Error adding background music: {e}")
        print("Continuing without background music")
        return final_video

def create_clip(media_path, asset_type, tts_path, duration=None, effects=None, narration_text=None, segment_index=0):
    """Create a video clip with synchronized subtitles and narration."""
    try:
        print(f"Creating clip #{segment_index} with asset_type: {asset_type}, media_path: {media_path}")
        if not os.path.exists(media_path) or not os.path.exists(tts_path):
            print("Missing media or TTS file")
            return None

        audio_clip = AudioFileClip(tts_path).audio_fadeout(0.2)
        audio_duration = audio_clip.duration
        target_duration = audio_duration + 0.2

        if asset_type == "video":
            clip = VideoFileClip(media_path)
            clip = resize_to_fill(clip, TARGET_RESOLUTION)
            if clip.duration < target_duration:
                clip = clip.loop(duration=target_duration)
            else:
                clip = clip.subclip(0, target_duration)
        elif asset_type == "image":
            img = Image.open(media_path)
            if img.mode != 'RGB':
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                    img.convert('RGB').save(temp.name)
                    media_path = temp.name
                img.close()
            clip = ImageClip(media_path).set_duration(target_duration)
            clip = apply_kenburns_effect(clip, TARGET_RESOLUTION)
            clip = clip.fadein(0.3).fadeout(0.3)
        else:
            return None

        # Skip subtitles for now since captions are disabled
        clip = clip.set_audio(audio_clip)
        print(f"Clip created: {clip.duration:.1f}s")
        return clip
    except Exception as e:
        print(f"Error in create_clip: {str(e)}")
        return None

# ---------------- Main Video Generation Function ---------------- #
def generate_video(user_input, resolution, caption_option):
    """Generate a video based on user input via Gradio."""
    global TARGET_RESOLUTION, CAPTION_COLOR, TEMP_FOLDER
    
    # Check if API keys are configured
    if not PEXELS_API_KEY or not OPENROUTER_API_KEY:
        return None
    
    # Set resolution
    if resolution == "Full":
        TARGET_RESOLUTION = (1920, 1080)
    elif resolution == "Short":
        TARGET_RESOLUTION = (1080, 1920)
    else:
        TARGET_RESOLUTION = (1920, 1080)  # Default

    # Set caption color
    CAPTION_COLOR = "white" if caption_option == "Yes" else "transparent"

    # Create a unique temporary folder
    TEMP_FOLDER = tempfile.mkdtemp()

    print("Generating script from API...")
    script = generate_script(user_input)
    if not script:
        print("Failed to generate script.")
        shutil.rmtree(TEMP_FOLDER)
        return None
    print("Generated Script:\n", script)
    elements = parse_script(script)
    if not elements:
        print("Failed to parse script into elements.")
        shutil.rmtree(TEMP_FOLDER)
        return None
    print(f"Parsed {len(elements)//2} script segments.")

    paired_elements = []
    for i in range(0, len(elements), 2):
        if i + 1 < len(elements):
            paired_elements.append((elements[i], elements[i + 1]))

    if not paired_elements:
        print("No valid script segments found.")
        shutil.rmtree(TEMP_FOLDER)
        return None

    clips = []
    for idx, (media_elem, tts_elem) in enumerate(paired_elements):
        print(f"\nProcessing segment {idx+1}/{len(paired_elements)} with prompt: '{media_elem['prompt']}'")
        media_asset = generate_media(media_elem['prompt'], current_index=idx, total_segments=len(paired_elements))
        if not media_asset:
            print(f"Skipping segment {idx+1} due to missing media asset.")
            continue
        tts_path = generate_tts(tts_elem['text'], tts_elem['voice'])
        if not tts_path:
            print(f"Skipping segment {idx+1} due to TTS generation failure.")
            continue
        clip = create_clip(
            media_path=media_asset['path'],
            asset_type=media_asset['asset_type'],
            tts_path=tts_path,
            duration=tts_elem['duration'],
            effects=media_elem.get('effects', 'fade-in'),
            narration_text=tts_elem['text'],
            segment_index=idx
        )
        if clip:
            clips.append(clip)
        else:
            print(f"Clip creation failed for segment {idx+1}.")

    if not clips:
        print("No clips were successfully created.")
        shutil.rmtree(TEMP_FOLDER)
        return None

    print("\nConcatenating clips...")
    final_video = concatenate_videoclips(clips, method="compose")
    final_video = add_background_music(final_video, bg_music_volume=bg_music_volume)

    print(f"Exporting final video to {OUTPUT_VIDEO_FILENAME}...")
    final_video.write_videofile(OUTPUT_VIDEO_FILENAME, codec='libx264', fps=fps, preset=preset)
    print(f"Final video saved as {OUTPUT_VIDEO_FILENAME}")

    # Clean up
    print("Cleaning up temporary files...")
    shutil.rmtree(TEMP_FOLDER)
    print("Temporary files removed.")

    return OUTPUT_VIDEO_FILENAME

# ---------------- Gradio Interface ---------------- #
VOICE_CHOICES = {
    'Emma (Female)': 'af_heart',
    'Bella (Female)': 'af_bella',
    'Nicole (Female)': 'af_nicole',
    'Aoede (Female)': 'af_aoede',
    'Kore (Female)': 'af_kore',
    'Sarah (Female)': 'af_sarah',
    'Nova (Female)': 'af_nova',
    'Sky (Female)': 'af_sky',
    'Alloy (Female)': 'af_alloy',
    'Jessica (Female)': 'af_jessica',
    'River (Female)': 'af_river',
    'Michael (Male)': 'am_michael',
    'Fenrir (Male)': 'am_fenrir',
    'Puck (Male)': 'am_puck',
    'Echo (Male)': 'am_echo',
    'Eric (Male)': 'am_eric',
    'Liam (Male)': 'am_liam',
    'Onyx (Male)': 'am_onyx',
    'Santa (Male)': 'am_santa',
    'Adam (Male)': 'am_adam',
    'Emma 🇬🇧 (Female)': 'bf_emma',
    'Isabella 🇬🇧 (Female)': 'bf_isabella',
    'Alice 🇬🇧 (Female)': 'bf_alice',
    'Lily 🇬🇧 (Female)': 'bf_lily',
    'George 🇬🇧 (Male)': 'bm_george',
    'Fable 🇬🇧 (Male)': 'bm_fable',
    'Lewis 🇬🇧 (Male)': 'bm_lewis',
    'Daniel 🇬🇧 (Male)': 'bm_daniel'
}

def generate_video_with_options(user_input, resolution, caption_option, music_file, voice, vclip_prob, bg_vol, video_fps, video_preset, v_speed, caption_size):
    global selected_voice, voice_speed, font_size, video_clip_probability, bg_music_volume, fps, preset
    
    # Check if API keys are configured
    if not PEXELS_API_KEY or not OPENROUTER_API_KEY:
        return gr.Error("API keys not configured. Please set PEXELS_API_KEY and OPENROUTER_API_KEY in Space secrets.")
    
    # Update global variables with user selections
    selected_voice = VOICE_CHOICES[voice]
    voice_speed = v_speed
    font_size = caption_size
    video_clip_probability = vclip_prob / 100  # Convert from percentage to decimal
    bg_music_volume = bg_vol
    fps = video_fps
    preset = video_preset
    
    # Handle music upload
    if music_file is not None:
        target_path = "music.mp3"
        shutil.copy(music_file.name, target_path)
        print(f"Uploaded music saved as: {target_path}")
    
    # Generate the video
    return generate_video(user_input, resolution, caption_option)

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_video_with_options,
    inputs=[
        gr.Textbox(label="Video Concept", placeholder="Enter your video concept here...", lines=3),
        gr.Radio(["Full", "Short"], label="Resolution", value="Full"),
        gr.Radio(["No"], label="Captions (Coming Soon)", value="No"),
        gr.File(label="Upload Background Music (MP3)", file_types=[".mp3"]),
        gr.Dropdown(choices=list(VOICE_CHOICES.keys()), label="Choose Voice", value="Emma (Female)"),
        gr.Slider(0, 100, value=25, step=1, label="Video Clip Usage Probability (%)"),
        gr.Slider(0.0, 1.0, value=0.08, step=0.01, label="Background Music Volume"),
        gr.Slider(10, 60, value=30, step=1, label="Video FPS"),
        gr.Dropdown(choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"], 
                   value="veryfast", label="Export Preset"),
        gr.Slider(0.5, 1.5, value=1.2, step=0.05, label="Voice Speed"),
        gr.Slider(20, 100, value=45, step=1, label="Caption Font Size")
    ],
    outputs=gr.Video(label="Generated Video"),
    title="AI Documentary Video Generator",
    description="Create short documentary videos with AI. Upload music, choose voice, and customize settings.\n\n⚠️ **Important**: Make sure to set your API keys in the Space secrets!"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()