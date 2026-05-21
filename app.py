# Import necessary libraries
import os
import re
import cv2
import math
import time
import random
import shutil
import io
import tempfile
import requests
import numpy as np
import soundfile as sf
import torch

from kokoro import KPipeline
from pydub import AudioSegment
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

# MoviePy integration
import moviepy.config as mpy_config
from moviepy.editor import (
    VideoFileClip, concatenate_videoclips, AudioFileClip, ImageClip,
    CompositeVideoClip, CompositeAudioClip, concatenate_audioclips
)
import moviepy.video.fx.all as vfx

import gradio as gr

# Initialize Kokoro TTS pipeline (American English configuration)
try:
    pipeline = KPipeline(lang_code='a')
except Exception as e:
    print(f"Warning initializing Kokoro Pipeline: {e}. Fallback systems active.")

# ---------------- Global Configuration ---------------- #

# Fetch environment API credentials safely
PEXELS_API_KEY = os.environ.get('PEXELS_API_KEY', '')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

# Local operational defaults
if not PEXELS_API_KEY:
    PEXELS_API_KEY = 'YOUR_PEXELS_KEY_HERE'
if not GROQ_API_KEY:
    GROQ_API_KEY = 'YOUR_GROQ_KEY_HERE'

GROQ_MODEL = "llama-3.3-70b-versatile"
OUTPUT_VIDEO_FILENAME = "final_video.mp4"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Execution state primitives
selected_voice = 'am_michael'
voice_speed = 1.0
font_size = 45
video_clip_probability = 0.65
bg_music_volume = 0.08
fps = 24
preset = "veryfast"
TARGET_RESOLUTION = (1920, 1080)
CAPTION_COLOR = "white"
TEMP_FOLDER = None

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

# ---------------- Core Support Functions ---------------- #

def check_api_keys():
    """Verify system infrastructure credentials are populated."""
    if not PEXELS_API_KEY or PEXELS_API_KEY == 'YOUR_PEXELS_KEY_HERE':
        return False, "PEXELS_API_KEY is not configured in environment variables."
    if not GROQ_API_KEY or GROQ_API_KEY == 'YOUR_GROQ_KEY_HERE':
        return False, "GROQ_API_KEY is not configured in environment variables."
    return True, "API keys configured successfully."

def generate_script(user_input):
    """Generate high-quality, humanized documentary scripts using the Groq API endpoint."""
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }

    prompt = f"""You are a witty, charismatic, and authentic human documentary narrator. Your task is to write an engaging video script based on the topic provided.

CRITICAL INSTRUCTIONS FOR NATURAL HUMAN TONE:
- Avoid ALL standard robotic AI tropes, filler terms, and clichés.
- Strictly DO NOT use words like: "delve", "tapestry", "testament", "furthermore", "moreover", "in conclusion", "look no further", "nestled", "beacon", or "revolutionize".
- Write with organic variety in sentence structure. Mix short, punchy statements with casual commentary.
- The humor should feel effortless, dry, and conversational—like a real person sharing funny, unexpected facts.

Format Requirements:
- Break the script into distinct scenes using structural brackets: [Tag].
- The Tag must be a simple 1-2 word search query suitable for finding background stock video/images (e.g., [Penguin], [City Traffic]).
- Directly beneath each tag, write exactly one engaging sentence (maximum 15 words) continuing the narrative.
- Conclude the piece with a [Subscribe] tag containing a clever, unexpected parting remark.

Topic: {user_input}
"""

    data = {
        'model': GROQ_MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.65,
        'max_tokens': 1500
    }

    try:
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=25
        )

        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content'].strip()
        else:
            print(f"Groq API Execution Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Network processing exception during script generation: {str(e)}")
        return None

def parse_script(script_text):
    """Robust regex parser extracting asset cues and voiceover text blocks."""
    if not script_text:
        return []
    
    # Matches tags in brackets and extracts all text leading up to the next bracket setup
    pattern = r'\[(.*?)\]\s*([^\[]+)'
    matches = re.findall(pattern, script_text)
    
    elements = []
    for tag, narration in matches:
        clean_tag = tag.strip()
        clean_narration = re.sub(r'\s+', ' ', narration.strip())
        
        if not clean_tag or not clean_narration:
            continue
            
        # Register video/visual search directive block
        elements.append({"type": "media", "prompt": clean_tag})
        
        # Determine temporal timing properties safely
        words = clean_narration.split()
        calculated_duration = max(3.5, len(words) * 0.45)
        elements.append({
            "type": "tts",
            "text": clean_narration,
            "duration": calculated_duration
        })
        
    return elements

def search_pexels_videos(query, pexels_api_key):
    """Query Pexels video directories for optimized landscape video content assets."""
    headers = {'Authorization': pexels_api_key}
    url = "https://api.pexels.com/videos/search"
    params = {"query": query, "per_page": 8, "orientation": "landscape"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=12)
        if response.status_code == 200:
            data = response.json()
            videos = data.get("videos", [])
            if videos:
                selected_video = random.choice(videos)
                video_files = selected_video.get("video_files", [])
                # Prioritize HD quality assets within typical delivery parameters
                for file in video_files:
                    if file.get("quality") == "hd" and file.get("width", 0) >= 1280:
                        return file.get("link")
                if video_files:
                    return video_files[0].get("link")
    except Exception as e:
        print(f"Pexels video fetch sequence exception: {e}")
    return None

def search_pexels_images(query, pexels_api_key):
    """Query Pexels asset directories for matching photographic components."""
    headers = {'Authorization': pexels_api_key}
    url = "https://api.pexels.com/v1/search"
    params = {"query": query, "per_page": 8, "orientation": "landscape"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=12)
        if response.status_code == 200:
            data = response.json()
            photos = data.get("photos", [])
            if photos:
                return random.choice(photos).get("src", {}).get("large2x")
    except Exception as e:
        print(f"Pexels image search exception lookup trace: {e}")
    return None

def download_asset_file(url, local_path):
    """Safely streams network binary assets into local storage blocks with validation context."""
    try:
        headers = {"User-Agent": USER_AGENT}
        with requests.get(url, headers=headers, stream=True, timeout=20) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
        return local_path
    except Exception as e:
        print(f"Error handling network asset download pipeline: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return None

def generate_solid_fallback(prompt, target_res):
    """Produces custom atmospheric placeholder graphics to prevent render failure workflows."""
    w, h = target_res
    random.seed(prompt)
    base_color = (random.randint(15, 35), random.randint(20, 40), random.randint(30, 55))
    
    img = Image.new('RGB', (w, h), base_color)
    fallback_path = os.path.join(TEMP_FOLDER, f"fallback_{int(time.time())}_{random.randint(0,99)}.jpg")
    img.save(fallback_path, quality=90)
    return fallback_path

def generate_media_asset(prompt):
    """Coordinates search matrices to fetch optimized video or graphic components."""
    safe_name = re.sub(r'[^\w\s-]', '', prompt).strip().replace(' ', '_')
    
    # Attempt high quality stock video stream compilation paths
    if random.random() < video_clip_probability:
        video_url = search_pexels_videos(prompt, PEXELS_API_KEY)
        if video_url:
            local_video_path = os.path.join(TEMP_FOLDER, f"vid_{safe_name}_{int(time.time())}.mp4")
            if download_asset_file(video_url, local_video_path):
                return {"path": local_video_path, "type": "video"}

    # Image processing search fallback tracks
    img_url = search_pexels_images(prompt, PEXELS_API_KEY)
    if img_url:
        local_img_path = os.path.join(TEMP_FOLDER, f"img_{safe_name}_{int(time.time())}.jpg")
        if download_asset_file(img_url, local_img_path):
            return {"path": local_img_path, "type": "image"}

    # Absolute safe fallback execution block
    fallback_image = generate_solid_fallback(prompt, TARGET_RESOLUTION)
    return {"path": fallback_image, "type": "image"}

def generate_tts_audio(text):
    """Converts script blocks into high quality WAV audio streams via Kokoro or gTTS fallbacks."""
    safe_name = re.sub(r'[^\w\s-]', '', text[:10]).strip().replace(' ', '_')
    output_audio_path = os.path.join(TEMP_FOLDER, f"tts_{safe_name}_{int(time.time())}.wav")

    try:
        # Pipeline generation handling through Kokoro system models
        generator = pipeline(text, voice=selected_voice, speed=voice_speed, split_pattern=r'\n+')
        audio_blocks = [audio for _, _, audio in generator]
        
        if audio_blocks:
            merged_audio = np.concatenate(audio_blocks) if len(audio_blocks) > 1 else audio_blocks[0]
            sf.write(output_audio_path, merged_audio, 24000)
            return output_audio_path
    except Exception as e:
        print(f"Kokoro engine synthesis bypass. Invoking alternative gTTS framework: {e}")
        
    try:
        # Alternative global synthesis path engines
        tts = gTTS(text=text, lang='en', slow=False)
        temp_mp3 = os.path.join(TEMP_FOLDER, f"gtts_{int(time.time())}.mp3")
        tts.save(temp_mp3)
        
        normalized_sound = AudioSegment.from_mp3(temp_mp3)
        normalized_sound.export(output_audio_path, format="wav")
        
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
        return output_audio_path
    except Exception as fail_err:
        print(f"Critical error: Synthesis framework down. Creating safe padding elements: {fail_err}")
        
        # Generate clean programmatic silence block to prevent pipeline execution crash
        duration_sec = max(3, len(text.split()) * 0.5)
        silent_samples = int(duration_sec * 24000)
        sf.write(output_audio_path, np.zeros(silent_samples, dtype=np.float32), 24000)
        return output_audio_path

def build_wrapped_subtitle_layer(text, canvas_resolution, font_size_target):
    """Calculates adaptive dynamic line wrapping parameters to print polished captions onto video matrices."""
    width, height = canvas_resolution
    max_text_boundary_width = int(width * 0.85)
    
    # Establish dynamic font structures safely across Linux/Windows hosting architectures
    font_engine = None
    font_options = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "/usr/share/fonts/Arial.ttf"
    ]
    for path in font_options:
        if os.path.exists(path):
            try:
                font_engine = ImageFont.truetype(path, font_size_target)
                break
            except:
                pass
    if font_engine is None:
        font_engine = ImageFont.load_default()

    # Form text line arrays matching specific horizontal screen boundaries
    words = text.split()
    compiled_lines = []
    current_line_build = []
    
    measurement_canvas = Image.new('RGBA', (1, 1))
    draw_inspector = ImageDraw.Draw(measurement_canvas)

    for word in words:
        test_string = " ".join(current_line_build + [word])
        try:
            box = draw_inspector.textbbox((0, 0), test_string, font=font_engine)
            calculated_w = box[2] - box[0]
        except:
            calculated_w = len(test_string) * (font_size_target * 0.55)
            
        if calculated_w <= max_text_boundary_width:
            current_line_build.append(word)
        else:
            if current_line_build:
                compiled_lines.append(" ".join(current_line_build))
            current_line_build = [word]
    if current_line_build:
        compiled_lines.append(" ".join(current_line_build))

    # Construct the graphic layout layer
    line_stride = font_size_target + 12
    computed_box_height = (len(compiled_lines) * line_stride) + 40
    
    subtitle_strip_canvas = Image.new('RGBA', (width, computed_box_height), (0, 0, 0, 0))
    context_drawer = ImageDraw.Draw(subtitle_strip_canvas)

    for idx, line in enumerate(compiled_lines):
        try:
            box = context_drawer.textbbox((0, 0), line, font=font_engine)
            w = box[2] - box[0]
        except:
            w = len(line) * (font_size_target * 0.55)
            
        target_x = (width - w) // 2
        target_y = 20 + (idx * line_stride)
        
        # High-contrast text layout shadow processing loops
        shadow_offsets = [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, -2), (0, 2), (-2, 0), (2, 0)]
        for dx, dy in shadow_offsets:
            context_drawer.text((target_x + dx, target_y + dy), line, font=font_engine, fill=(0, 0, 0, 225))
            
        # Draw text top presentation layer
        context_drawer.text((target_x, target_y), line, font=font_engine, fill=(255, 255, 255, 255))

    local_subtitle_png_path = os.path.join(TEMP_FOLDER, f"sub_{int(time.time())}_{random.randint(100,999)}.png")
    subtitle_strip_canvas.save(local_subtitle_png_path)
    return local_subtitle_png_path

def resize_and_crop_to_fill(clip, target_resolution):
    """Crops background tracks seamlessly to fit target composition frames without aspect stretching distortion."""
    tw, th = target_resolution
    clip_w, clip_h = clip.w, clip.h
    
    aspect_target = tw / th
    aspect_clip = clip_w / clip_h
    
    if aspect_clip > aspect_target:
        scale_factor = th / clip_h
        scaled_w = int(clip_w * scale_factor)
        resized_clip = clip.resize(newsize=(scaled_w, th))
        crop_start_x = (scaled_w - tw) / 2
        return resized_clip.crop(x1=crop_start_x, x2=crop_start_x + tw, y1=0, y2=th)
    else:
        scale_factor = tw / clip_w
        scaled_h = int(clip_h * scale_factor)
        resized_clip = clip.resize(newsize=(tw, scaled_h))
        crop_start_y = (scaled_h - th) / 2
        return resized_clip.crop(x1=0, x2=tw, y1=crop_start_y, y2=crop_start_y + th)

def apply_cinematic_motion_effect(clip, target_resolution):
    """Injects steady cinematic animation tracking routines over static photo frames."""
    tw, th = target_resolution
    
    # Scale canvas boundaries slightly to establish buffer tracks for pan scanning operations
    base_clip = clip.resize(newsize=(int(tw * 1.2), int(th * 1.2)))
    max_delta_x = base_clip.w - tw
    max_delta_y = base_clip.h - th

    motion_style = random.choice(["zoom-in", "zoom-out", "pan-right", "pan-left"])
    clip_duration = clip.duration

    def frame_transformation_matrix(get_frame, t):
        frame = get_frame(t)
        progress = (t / clip_duration) if clip_duration > 0 else 0
        # Smooth sinusoidal mapping curves
        smooth_step = 0.5 * (1.0 - math.cos(math.pi * progress))
        
        if motion_style == "zoom-in":
            scale_current = 1.0 + (0.12 * smooth_step)
            curr_w, curr_h = int(tw * scale_current), int(th * scale_current)
            resized_f = cv2.resize(frame, (curr_w, curr_h), interpolation=cv2.INTER_LINEAR)
            x_start = (curr_w - tw) // 2
            y_start = (curr_h - th) // 2
            return resized_f[y_start:y_start+th, x_start:x_start+tw]
            
        elif motion_style == "zoom-out":
            scale_current = 1.15 - (0.12 * smooth_step)
            curr_w, curr_h = int(tw * scale_current), int(th * scale_current)
            resized_f = cv2.resize(frame, (curr_w, curr_h), interpolation=cv2.INTER_LINEAR)
            x_start = (curr_w - tw) // 2
            y_start = (curr_h - th) // 2
            return resized_f[y_start:y_start+th, x_start:x_start+tw]
            
        elif motion_style == "pan-right":
            x_offset = int(max_delta_x * smooth_step)
            y_center = max_delta_y // 2
            return frame[y_center:y_center+th, x_offset:x_offset+tw]
            
        else: # pan-left
            x_offset = int(max_delta_x * (1.0 - smooth_step))
            y_center = max_delta_y // 2
            return frame[y_center:y_center+th, x_offset:x_offset+tw]

    return base_clip.fl(frame_transformation_matrix)

def compile_individual_segment(media_path, asset_type, audio_path, script_text, segment_id):
    """Combines voice, subtitles, and background visual elements into a single composite sequence block."""
    video_sequence_clip = None
    voice_track_clip = None
    subtitle_overlay_clip = None
    composite_block = None
    
    try:
        if not os.path.exists(media_path) or not os.path.exists(audio_path):
            return None

        voice_track_clip = AudioFileClip(audio_path).fx(vfx.audio_fadeout, 0.15)
        allocated_duration = voice_track_clip.duration + 0.35

        if asset_type == "video":
            raw_video = VideoFileClip(media_path, audio=False)
            normalized_video = resize_and_crop_to_fill(raw_video, TARGET_RESOLUTION)
            
            if normalized_video.duration < allocated_duration:
                video_sequence_clip = normalized_video.fx(vfx.loop, duration=allocated_duration)
                raw_video.close()
                normalized_video.close()
            else:
                video_sequence_clip = normalized_video.subclip(0, allocated_duration)
                raw_video.close()
                normalized_video.close()
        else:
            base_image = ImageClip(media_path).set_duration(allocated_duration)
            animated_image = apply_cinematic_motion_effect(base_image, TARGET_RESOLUTION)
            video_sequence_clip = animated_image.fx(vfx.fadein, 0.25).fx(vfx.fadeout, 0.25)
            base_image.close()

        video_sequence_clip = video_sequence_clip.set_audio(voice_track_clip)

        # Append caption overlays dynamically if requested
        if CAPTION_COLOR == "white" and script_text:
            subtitle_image_file = build_wrapped_subtitle_layer(script_text, TARGET_RESOLUTION, font_size)
            subtitle_overlay_clip = (ImageClip(subtitle_image_file)
                                     .set_duration(allocated_duration)
                                     .set_position(('center', int(TARGET_RESOLUTION[1] * 0.78))))
            composite_block = CompositeVideoClip([video_sequence_clip, subtitle_overlay_clip], size=TARGET_RESOLUTION)
            return composite_block
        else:
            return video_sequence_clip

    except Exception as err:
        print(f"Exception encountered while processing sequence segment block assembly #{segment_id}: {err}")
        # Clean execution leaks
        try:
            if video_sequence_clip: video_sequence_clip.close()
            if voice_track_clip: voice_track_clip.close()
            if subtitle_overlay_clip: subtitle_overlay_clip.close()
        except:
            pass
        return None

# ---------------- Pipeline Processing Entry ---------------- #

def generate_video(user_input, resolution_mode, enable_captions):
    """Main orchestrator function that controls script generation, file handling, asset searches, and rendering."""
    global TARGET_RESOLUTION, CAPTION_COLOR, TEMP_FOLDER
    
    api_ok, check_msg = check_api_keys()
    if not api_ok:
        return None, f"Configuration Error: {check_msg}"
        
    TARGET_RESOLUTION = (1920, 1080) if resolution_mode == "Full (16:9)" else (1080, 1920)
    CAPTION_COLOR = "white" if enable_captions == "Yes" else "transparent"
    TEMP_FOLDER = tempfile.mkdtemp()
    
    master_clip_list = []
    final_output_composition = None

    try:
        print("Contacting Groq processing endpoints for narrative compilation...")
        generated_script_text = generate_script(user_input)
        
        if not generated_script_text:
            return None, "Failed to retrieve processing directives from Groq script modules."
            
        print(f"\n--- Output Script Generated ---\n{generated_script_text}\n-------------------------------")
        
        script_execution_steps = parse_script(generated_script_text)
        if not script_execution_steps:
            return None, "Parser structural error. Could not read tags from the generated script structure."

        # Connect matching narrative scripts to corresponding scene prompts
        paired_segments = []
        for i in range(0, len(script_execution_steps), 2):
            if i + 1 < len(script_execution_steps):
                paired_segments.append((script_execution_steps[i], script_execution_steps[i+1]))

        print(f"Beginning rendering sequences for {len(paired_segments)} active scenes.")

        for idx, (media_cue, audio_cue) in enumerate(paired_segments):
            print(f"Rendering scene progression track ({idx + 1}/{len(paired_segments)}) -> Tag: {media_cue['prompt']}")
            
            media_asset = generate_media_asset(media_cue['prompt'])
            audio_track_file = generate_tts_audio(audio_cue['text'])
            
            constructed_segment = compile_individual_segment(
                media_path=media_asset['path'],
                asset_type=media_asset['type'],
                audio_path=audio_track_file,
                script_text=audio_cue['text'],
                segment_id=idx
            )
            
            if constructed_segment:
                master_clip_list.append(constructed_segment)

        if not master_clip_list:
            return None, "Pipeline Error: Could not successfully render any independent scene tracks."

        print("Assembling timeline segments into master stream...")
        final_output_composition = concatenate_videoclips(master_clip_list, method="compose")

        # Mix background audio files if available
        bg_music_source = "music.mp3"
        if os.path.exists(bg_music_source):
            try:
                print("Injecting background music matrix layer...")
                ambient_music_clip = AudioFileClip(bg_music_source)
                
                if ambient_music_clip.duration < final_output_composition.duration:
                    loop_iterations = math.ceil(final_output_composition.duration / ambient_music_clip.duration)
                    ambient_music_clip = concatenate_audioclips([ambient_music_clip] * loop_iterations)
                    
                ambient_music_clip = (ambient_music_clip
                                      .subclip(0, final_output_composition.duration)
                                      .fx(vfx.volumex, bg_music_volume))
                
                mixed_audio_output = CompositeAudioClip([final_output_composition.audio, ambient_music_clip])
                final_output_composition = final_output_composition.set_audio(mixed_audio_output)
            except Exception as music_err:
                print(f"Background ambient track mixing exception bypassed: {music_err}")

        print(f"Encoding final MP4 video stream layer ({fps} FPS)...")
        final_output_composition.write_videofile(
            OUTPUT_VIDEO_FILENAME,
            codec='libx264',
            audio_codec='aac',
            fps=fps,
            preset=preset,
            threads=4,
            logger=None
        )
        
        return OUTPUT_VIDEO_FILENAME, "Cinematic video generation completed successfully!"

    except Exception as pipeline_fault:
        print(f"Critical execution failure inside video generation pipeline: {pipeline_fault}")
        return None, f"Execution Failure: {str(pipeline_fault)}"
        
    finally:
        print("Executing memory storage cleanup tracks...")
        try:
            if final_output_composition:
                final_output_composition.close()
            for structural_clip in master_clip_list:
                if structural_clip:
                    structural_clip.close()
        except Exception as close_err:
            print(f"Resource lock release trace warning: {close_err}")
            
        time.sleep(1.2)
        if TEMP_FOLDER and os.path.exists(TEMP_FOLDER):
            try:
                shutil.rmtree(TEMP_FOLDER)
                print("Temporary working directory flushed successfully.")
            except Exception as flush_err:
                print(f"Warning clearing temporary directory blocks: {flush_err}")

# ---------------- Gradio Interface Binding Maps ---------------- #

def UI_interaction_bridge(prompt, res_mode, caption_flag, music_upload, voice, v_prob, music_vol, frames_per_sec, speed_preset, tts_speed, size_font):
    """Synchronizes UI settings variables with background script parameters."""
    global selected_voice, voice_speed, font_size, video_clip_probability, bg_music_volume, fps, preset
    
    selected_voice = VOICE_CHOICES.get(voice, 'am_michael')
    voice_speed = tts_speed
    font_size = size_font
    video_clip_probability = v_prob / 100.0
    bg_music_volume = music_vol
    fps = frames_per_sec
    preset = speed_preset
    
    if music_upload is not None:
        destination_path = "music.mp3"
        try:
            shutil.copy(music_upload.name, destination_path)
            print(f"New ambient track file successfully mounted: {destination_path}")
        except Exception as e:
            print(f"Error mounting audio file resource track: {e}")
            
    return generate_video(prompt, res_mode, caption_flag)

# Constructing layout structures
with gr.Blocks(title="AI Cinematic Documentary Generator") as app_interface:
    gr.Markdown("# 🎬 AI Cinematic Documentary Video Generator")
    gr.Markdown("Instantly build automated documentary videos fueled by high-performance Groq Llama-3.3 intelligence and automated stock footage tracking matrices.")
    
    with gr.Row():
        with gr.Column(scale=1):
            user_prompt_input = gr.Textbox(
                label="Documentary Video Concept / Prompt",
                placeholder="Ex: The secret comedic lives of household cats when owners go to work...",
                lines=4
            )
            
            with gr.Row():
                ui_resolution = gr.Radio(["Full (16:9)", "Shorts (9:16)"], label="Video Output Dimension", value="Full (16:9)")
                ui_captions = gr.Radio(["Yes", "No"], label="Render Subtitle Captions", value="Yes")
            
            ui_audio_upload = gr.File(label="Optional Background Music Tracking (MP3 Only)", file_types=[".mp3"])
            
            with gr.Accordion("Advanced Cinema Tuning Configuration", open=False):
                ui_voice_selection = gr.Dropdown(
                    choices=list(VOICE_CHOICES.keys()),
                    label="Narration Voice Model Selection",
                    value="Michael (Male)"
                )
                ui_video_probability = gr.Slider(0, 100, value=70, step=5, label="Target Video vs. Photo Probability (%)")
                ui_music_level = gr.Slider(0.00, 0.40, value=0.06, step=0.01, label="Background Music Volume Mix Factor")
                ui_fps_target = gr.Slider(15, 60, value=24, step=1, label="Target Output Encoding FPS")
                ui_encoding_speed = gr.Dropdown(
                    choices=["ultrafast", "superfast", "veryfast", "medium", "slow"],
                    value="veryfast",
                    label="FFmpeg Encoding Pipeline Speed Preset"
                )
                ui_voice_tempo = gr.Slider(0.6, 1.4, value=1.05, step=0.05, label="Narration Voice Pace Speed")
                ui_caption_font_size = gr.Slider(20, 90, value=48, step=2, label="Caption Title Text Size Scaling")
            
            trigger_generation_button = gr.Button("🎬 Render Cinematic Video Sequence", variant="primary")
            
        with gr.Column(scale=1):
            video_output_display = gr.Video(label="Final Render Output Component Preview")
            pipeline_status_log = gr.Textbox(label="System Operational Pipeline Logs", interactive=False)
            
    gr.Markdown("""
    ### ⚙️ Production Guidelines & Prerequisites:
    1. Verify your operational environment contains verified **GROQ_API_KEY** and **PEXELS_API_KEY** secret strings.
    2. Optional backing tracks look for or utilize uploaded `.mp3` files mapped directly into working context spaces.
    3. The subtitle caption generator functions natively on a zero-dependency Pillow overlay architecture to maximize compatibility with Hugging Face deployment spaces.
    """)

    trigger_generation_button.click(
        fn=UI_interaction_bridge,
        inputs=[
            user_prompt_input, ui_resolution, ui_captions, ui_audio_upload,
            ui_voice_selection, ui_video_probability, ui_music_level, ui_fps_target,
            ui_encoding_speed, ui_voice_tempo, ui_caption_font_size
        ],
        outputs=[video_output_display, pipeline_status_log]
    )

if __name__ == "__main__":
    app_interface.launch()