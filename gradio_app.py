import os
import sys
import logging
import re
import gradio as gr
from dotenv import load_dotenv
import uuid

# Add the current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import modules
from media_ingestion import MediaIngester
from speech_recognition import SpeechRecognizer
from speech_diarization import SpeakerDiarizer
from translate import translate_text, generate_srt_subtitles
from text_to_speech import generate_tts
from audio_to_video import create_video_with_mixed_audio

# Configuration
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories
os.makedirs("temp", exist_ok=True)
os.makedirs("audio", exist_ok=True)
os.makedirs("audio2", exist_ok=True)
os.makedirs("reference_audio", exist_ok=True)

# Global state
processing_status = {}

def create_session_id():
    return str(uuid.uuid4())[:8]

def process_video(media_source, target_language, tts_choice, max_speakers, speaker_config, session_id, progress=gr.Progress()):
    try:
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            return {"error": "Missing HUGGINGFACE_TOKEN"}
        
        progress(0.05, desc="Initializing")
        ingester = MediaIngester("temp")
        recognizer = SpeechRecognizer("base")
        diarizer = SpeakerDiarizer(hf_token)

        progress(0.1, desc="Processing media")
        video_path = ingester.process_input(media_source)
        audio_path = ingester.extract_audio(video_path)

        progress(0.15, desc="Cleaning audio")
        clean_audio, bg_audio = ingester.separate_audio_sources(audio_path)

        progress(0.2, desc="Transcribing")
        segments = recognizer.transcribe(clean_audio)

        progress(0.3, desc="Identifying speakers")
        max_spk = int(max_speakers) if max_speakers.strip() else None
        speakers = diarizer.diarize(clean_audio, max_spk)

        progress(0.4, desc="Assigning speakers")
        final_segments = diarizer.assign_speakers_to_segments(segments, speakers)

        progress(0.5, desc="Translating")
        translated = translate_text(final_segments, target_language)
        subtitle_file = f"temp/{os.path.basename(video_path).split('.')[0]}_{target_language}.srt"
        generate_srt_subtitles(translated, subtitle_file)

        progress(0.6, desc="Configuring voices")
        unique_speakers = {seg['speaker'] for seg in translated if 'speaker' in seg}
        voice_config = {}
        use_xtts = tts_choice == "Voice cloning (XTTS)"

        if use_xtts:
            ref_files = diarizer.extract_speaker_references(clean_audio, speakers, "reference_audio")
            for speaker in unique_speakers:
                if match := re.match(r"SPEAKER_(\d+)", speaker):
                    spk_id = int(match.group(1))
                    voice_config[spk_id] = {
                        'engine': 'xtts' if speaker in ref_files else 'edge_tts',
                        'reference_audio': ref_files.get(speaker),
                        'gender': speaker_config.get(str(spk_id), "female"),
                        'language': target_language
                    }
        else:
            for speaker in unique_speakers:
                if match := re.match(r"SPEAKER_(\d+)", speaker):
                    spk_id = int(match.group(1))
                    voice_config[spk_id] = {
                        'engine': 'edge_tts',
                        'gender': speaker_config.get(str(spk_id), "female")
                    }

        progress(0.7, desc="Generating TTS")
        dubbed_audio = generate_tts(translated, target_language, voice_config, "audio2")

        progress(0.85, desc="Mixing audio")
        output_path = create_video_with_mixed_audio(video_path, bg_audio, dubbed_audio)

        return {
            "video": output_path,
            "subtitle": subtitle_file,
            "message": "Success!"
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            "video": None,
            "subtitle": None,
            "message": f"Error: {str(e)}"
        }

def create_interface():
    with gr.Blocks(title="SyncDub", theme=gr.themes.Soft()) as app:
        gr.Markdown("# SyncDub - Video Translation & Dubbing")
        session_id = gr.State(create_session_id)
        
        with gr.Row():
            with gr.Column():
                media = gr.Textbox(label="Video URL/Path")
                upload = gr.File(label="Or Upload Video", file_types=["video"])
                lang = gr.Dropdown(
                    choices=["en", "es", "hi" ,"fr", "de", "it", "ja", "ko", "pt", "ru", "zh"],
                    label="Target Language",
                    value="en"
                )
                tts = gr.Radio(
                    choices=["Simple dubbing (Edge TTS)", "Voice cloning (XTTS)"],
                    label="TTS Method",
                    value="Simple dubbing (Edge TTS)"
                )
                speakers = gr.Slider(1, 8, value=1, step=1, label="Max Speakers")
                inputs = gr.Column()
                btn = gr.Button("Process", variant="primary")
                status = gr.Textbox(label="Status")
            
            with gr.Column():
                video = gr.Video(label="Output")
                subs = gr.File(label="Subtitles")
                msg = gr.Textbox(label="Message")

        def update_inputs(tts_method, spk_count):
            components = []
            for i in range(int(spk_count)):
                if tts_method == "Simple dubbing (Edge TTS)":
                    components.append(gr.Radio(
                        ["male", "female"], 
                        label=f"Speaker {i+1} Gender",
                        value="female"
                    ))
                else:
                    components.append(gr.File(
                        label=f"Speaker {i+1} Reference",
                        file_types=["audio"]
                    ))
            return components

        tts.change(update_inputs, [tts, speakers], inputs)
        speakers.change(update_inputs, [tts, speakers], inputs)

        def wrapper(media_url, media_file, lang, tts, spk, *inputs):
            source = media_file or media_url
            config = {str(i): val for i, val in enumerate(inputs) if val}
            return process_video(source, lang, tts, spk, config, session_id.value)
        
        btn.click(
            wrapper,
            [media, upload, lang, tts, speakers, inputs],
            [video, subs, msg]
        )
        
        # Timer with correct parameter for latest Gradio versions
        timer = gr.Timer(
            interval=1,  # Works with newer versions
            fn=lambda: processing_status.get(session_id.value, {}).get("status", "Ready"),
            outputs=status
        )

    return app

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)