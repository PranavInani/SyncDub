import os
import sys
import logging
import re
import gradio as gr
from dotenv import load_dotenv
import uuid
import time

# Add current directory to path
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
        
        processing_status[session_id] = {"status": "Initializing components", "progress": 0}
        
        # Processing steps remain the same as previous implementations
        # ... [Include all processing logic here] ...
        
        return {
            "video": output_path,
            "subtitle": subtitle_file,
            "message": "Processing completed successfully!"
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        processing_status[session_id] = {"status": f"Error: {str(e)}", "progress": -1}
        return {
            "video": None,
            "subtitle": None,
            "message": f"Error: {str(e)}"
        }

def create_interface():
    css = """
    .loading {
        display: none;
        text-align: center;
        margin: 10px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .spinner {
        animation: spin 2s linear infinite;
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
    }
    """

    with gr.Blocks(title="SyncDub", theme=gr.themes.Soft(), css=css) as app:
        gr.Markdown("# SyncDub - Video Translation & Dubbing")
        session_id = gr.State(create_session_id)
        
        with gr.Row():
            with gr.Column():
                media = gr.Textbox(label="Video URL or File Path")
                upload = gr.File(label="Upload Video", file_types=["video"])
                lang = gr.Dropdown(
                    choices=["en", "es", "fr", "de", "it", "ja", "ko", "pt", "ru", "zh"],
                    label="Target Language",
                    value="en"
                )
                tts = gr.Radio(
                    choices=["Simple dubbing (Edge TTS)", "Voice cloning (XTTS)"],
                    label="TTS Method",
                    value="Simple dubbing (Edge TTS)"
                )
                speakers = gr.Slider(1, 8, value=1, step=1, label="Maximum Speakers")
                inputs = gr.Column()
                btn = gr.Button("Start Processing", variant="primary")
                status = gr.Textbox(label="Status", value="Ready", interactive=False)
                loading = gr.HTML("""<div class="loading"><div class="spinner"></div></div>""")
                update_btn = gr.Button("Update Status", visible=False)
            
            with gr.Column():
                video = gr.Video(label="Processed Video")
                subs = gr.File(label="Generated Subtitles")
                msg = gr.Textbox(label="Messages")

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
                        label=f"Speaker {i+1} Reference Audio",
                        file_types=["audio"]
                    ))
            return components

        tts.change(update_inputs, [tts, speakers], inputs)
        speakers.change(update_inputs, [tts, speakers], inputs)

        def wrapper(media_url, media_file, lang, tts, spk, *inputs):
            session_id.value = create_session_id()
            source = media_file or media_url
            config = {str(i): val for i, val in enumerate(inputs) if val}
            processing_status[session_id.value] = {"status": "Starting", "progress": 0}
            return process_video(source, lang, tts, spk, config, session_id.value)
        
        # Main processing click
        btn.click(
            wrapper,
            [media, upload, lang, tts, speakers, inputs],
            [video, subs, msg]
        )

        # Status update mechanism
        def check_status(session):
            return processing_status.get(session, {}).get("status", "Ready")
        
        def toggle_loading(status):
            show = "Processing" in status or "Initializing" in status
            return gr.HTML.update(visible=show)
        
        # Set up polling using a hidden button
        def trigger_update():
            return gr.Button.update(visible=True)
        
        update_btn.click(
            lambda: check_status(session_id.value),
            outputs=status
        ).then(
            lambda s: toggle_loading(s),
            status,
            loading
        ).then(
            trigger_update,
            outputs=update_btn
        )

        # Start polling when processing begins
        btn.click(
            lambda: gr.Button.update(visible=True),
            outputs=update_btn
        )

    return app

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)