import os
import sys
import logging
import re
import gradio as gr
from dotenv import load_dotenv
import uuid

# Add the current directory to path to help with imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import required modules
from media_ingestion import MediaIngester
from speech_recognition import SpeechRecognizer
from speech_diarization import SpeakerDiarizer
from translate import translate_text, generate_srt_subtitles
from text_to_speech import generate_tts
from audio_to_video import create_video_with_mixed_audio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("temp", exist_ok=True)
os.makedirs("audio", exist_ok=True)
os.makedirs("audio2", exist_ok=True)
os.makedirs("reference_audio", exist_ok=True)

# Global variables for process tracking
processing_status = {}

def create_session_id():
    """Create a unique session ID"""
    return str(uuid.uuid4())[:8]

def process_video(media_source, target_language, tts_choice, max_speakers, speaker_config, session_id, progress=gr.Progress()):
    """Main processing pipeline"""
    global processing_status
    processing_status[session_id] = {"status": "Starting", "progress": 0}
    
    try:
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            return {"error": "Missing HUGGINGFACE_TOKEN in .env file"}
        
        # Initialize components
        progress(0.05, desc="Initializing components")
        ingester = MediaIngester(output_dir="temp")
        recognizer = SpeechRecognizer(model_size="base")
        diarizer = SpeakerDiarizer(hf_token=hf_token)

        # Process media input
        progress(0.1, desc="Processing media source")
        video_path = ingester.process_input(media_source)
        audio_path = ingester.extract_audio(video_path)

        # Audio processing
        progress(0.15, desc="Separating audio sources")
        clean_audio_path, bg_audio_path = ingester.separate_audio_sources(audio_path)

        # Speech recognition
        progress(0.2, desc="Transcribing audio")
        segments = recognizer.transcribe(clean_audio_path)

        # Speaker diarization
        progress(0.3, desc="Identifying speakers")
        max_speakers_val = int(max_speakers) if max_speakers and max_speakers.strip() else None
        speakers = diarizer.diarize(clean_audio_path, max_speakers=max_speakers_val)

        # Assign speakers
        progress(0.4, desc="Assigning speakers to segments")
        final_segments = diarizer.assign_speakers_to_segments(segments, speakers)

        # Translation
        progress(0.5, desc=f"Translating to {target_language}")
        translated_segments = translate_text(final_segments, target_lang=target_language)

        # Generate subtitles
        subtitle_file = f"temp/{os.path.basename(video_path).split('.')[0]}_{target_language}.srt"
        generate_srt_subtitles(translated_segments, output_file=subtitle_file)

        # Voice configuration
        progress(0.6, desc="Configuring voices")
        unique_speakers = set(seg.get('speaker') for seg in translated_segments if 'speaker' in seg)
        voice_config = {}
        use_voice_cloning = tts_choice == "Voice cloning (XTTS)"

        if use_voice_cloning:
            reference_files = diarizer.extract_speaker_references(clean_audio_path, speakers, "reference_audio")
            for speaker in unique_speakers:
                if match := re.match(r"SPEAKER_(\d+)", speaker):
                    spk_id = int(match.group(1))
                    if speaker in reference_files:
                        voice_config[spk_id] = {
                            'engine': 'xtts',
                            'reference_audio': reference_files[speaker],
                            'language': target_language
                        }
                    else:
                        voice_config[spk_id] = {
                            'engine': 'edge_tts',
                            'gender': speaker_config.get(str(spk_id), "female")
                        }
        else:
            for speaker in unique_speakers:
                if match := re.match(r"SPEAKER_(\d+)", speaker):
                    spk_id = int(match.group(1))
                    voice_config[spk_id] = {
                        'engine': 'edge_tts',
                        'gender': speaker_config.get(str(spk_id), "female")
                    }

        # Generate TTS
        progress(0.7, desc="Generating speech")
        dubbed_audio_path = generate_tts(translated_segments, target_language, voice_config, "audio2")

        # Create final video
        progress(0.85, desc="Creating final video")
        output_path = create_video_with_mixed_audio(video_path, bg_audio_path, dubbed_audio_path)

        progress(1.0, desc="Process completed")
        return {
            "video": output_path,
            "subtitle": subtitle_file,
            "message": "Process completed successfully!"
        }
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return {
            "video": None,
            "subtitle": None,
            "message": f"Error: {str(e)}"
        }

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="SyncDub - Video Translation", theme=gr.themes.Soft()) as app:
        gr.Markdown("# SyncDub - Video Translation & Dubbing")
        
        # Session management
        session_id = gr.State(value=create_session_id)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Media input
                media_source = gr.File(label="Upload Video", file_types=["video"], type="filepath")
                media_url = gr.Textbox(label="or Video URL", placeholder="Enter YouTube/Media URL")
                
                # Settings
                target_language = gr.Dropdown(
                    label="Target Language",
                    choices=["en","hi", "es", "fr", "de", "it", "ja", "ko", "pt", "ru", "zh"],
                    value="en"
                )
                tts_method = gr.Radio(
                    label="Dubbing Method",
                    choices=["Simple dubbing (Edge TTS)", "Voice cloning (XTTS)"],
                    value="Simple dubbing (Edge TTS)"
                )
                max_speakers = gr.Slider(1, 8, value=1, step=1, label="Max Speakers (0=auto)")
                
                # Dynamic speaker inputs
                speaker_inputs = gr.Column()
                
                # Process button
                process_btn = gr.Button("Start Processing", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=2):
                # Outputs
                output_video = gr.Video(label="Dubbed Video")
                output_subtitles = gr.File(label="Generated Subtitles")
                output_message = gr.Textbox(label="Messages")

        # Dynamic speaker configuration
        def update_speaker_inputs(tts_choice, speaker_count):
            inputs = []
            for i in range(int(speaker_count)):
                if tts_choice == "Simple dubbing (Edge TTS)":
                    inputs.append(gr.Radio(
                        choices=["male", "female"],
                        label=f"Speaker {i+1} Gender",
                        value="female"
                    ))
                else:
                    inputs.append(gr.File(
                        label=f"Speaker {i+1} Reference Audio",
                        file_types=["audio"]
                    ))
            return inputs

        tts_method.change(
            update_speaker_inputs,
            [tts_method, max_speakers],
            speaker_inputs
        )
        
        max_speakers.change(
            update_speaker_inputs,
            [tts_method, max_speakers],
            speaker_inputs
        )

        # Processing logic
        def process_wrapper(media_url, media_upload, target_lang, tts_method, max_spk, *spk_inputs):
            # Combine media inputs
            media_source = media_upload or media_url
            
            # Create speaker config
            speaker_config = {}
            for idx, input_val in enumerate(spk_inputs):
                if isinstance(input_val, dict):  # File input
                    speaker_config[str(idx)] = input_val.get("name", "")
                elif input_val in ["male", "female"]:
                    speaker_config[str(idx)] = input_val
            
            return process_video(
                media_source, target_lang, tts_method, max_spk,
                speaker_config, session_id.value
            )

        # Connect components
        process_btn.click(
            process_wrapper,
            [media_url, media_source, target_language, tts_method, max_speakers, speaker_inputs],
            [output_video, output_subtitles, output_message]
        )
        
        # Status updates
        status_timer = gr.Timer(
            every=1,
            fn=lambda: processing_status.get(session_id.value, {}).get("status", "Ready"),
            outputs=status
        )

    return app

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)