import os
import sys
import logging
import tempfile
import re
import gradio as gr
from dotenv import load_dotenv
import threading
import uuid

# Add the current directory to path to help with imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the required modules
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
    """Create a unique session ID for tracking progress"""
    return str(uuid.uuid4())[:8]

def process_video(media_source, target_language, tts_choice, max_speakers, speaker_config, session_id, progress=gr.Progress()):
    """Main processing function that handles the complete pipeline"""
    global processing_status
    processing_status[session_id] = {"status": "Starting", "progress": 0}
    
    try:
        # Get API tokens
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        if not hf_token:
            return {"error": "HUGGINGFACE_TOKEN not found in .env file. Please set it up."}
        
        # Initialize components
        progress(0.05, desc="Initializing components")
        processing_status[session_id] = {"status": "Initializing components", "progress": 0.05}
        
        ingester = MediaIngester(output_dir="temp")
        recognizer = SpeechRecognizer(model_size="base")
        diarizer = SpeakerDiarizer(hf_token=hf_token)
        
        # Step 1: Process input and extract audio
        progress(0.1, desc="Processing media source")
        processing_status[session_id] = {"status": "Processing media source", "progress": 0.1}
        
        video_path = ingester.process_input(media_source)
        audio_path = ingester.extract_audio(video_path)
        
        progress(0.15, desc="Separating audio sources")
        processing_status[session_id] = {"status": "Separating audio sources", "progress": 0.15}
        
        clean_audio_path, bg_audio_path = ingester.separate_audio_sources(audio_path)
        
        # Step 2: Perform speech recognition
        progress(0.2, desc="Transcribing audio")
        processing_status[session_id] = {"status": "Transcribing audio", "progress": 0.2}
        
        segments = recognizer.transcribe(clean_audio_path)
        
        # Step 3: Perform speaker diarization
        progress(0.3, desc="Identifying speakers")
        processing_status[session_id] = {"status": "Identifying speakers", "progress": 0.3}
        
        max_speakers_val = int(max_speakers) if max_speakers and max_speakers.strip() else None
        speakers = diarizer.diarize(clean_audio_path, max_speakers=max_speakers_val)
        
        # Step 4: Assign speakers to segments
        progress(0.4, desc="Assigning speakers to segments")
        processing_status[session_id] = {"status": "Assigning speakers to segments", "progress": 0.4}
        
        final_segments = diarizer.assign_speakers_to_segments(segments, speakers)
        
        # Step 5: Translate the segments
        progress(0.5, desc=f"Translating to {target_language}")
        processing_status[session_id] = {"status": f"Translating to {target_language}", "progress": 0.5}
        
        translated_segments = translate_text(
            final_segments, 
            target_lang=target_language,
            translation_method="batch"
        )
        
        subtitle_file = f"temp/{os.path.basename(video_path).split('.')[0]}_{target_language}.srt"
        generate_srt_subtitles(translated_segments, output_file=subtitle_file)
        
        # Step 6: Configure voice characteristics
        progress(0.6, desc="Configuring voices")
        processing_status[session_id] = {"status": "Configuring voices", "progress": 0.6}
        
        unique_speakers = set(segment.get('speaker') for segment in translated_segments if 'speaker' in segment)
        voice_config = {}
        use_voice_cloning = tts_choice == "Voice cloning (XTTS)"
        
        if use_voice_cloning:
            reference_files = diarizer.extract_speaker_references(
                clean_audio_path, 
                speakers, 
                output_dir="reference_audio"
            )
            
            for speaker in unique_speakers:
                if match := re.search(r'SPEAKER_(\d+)', speaker):
                    speaker_id = int(match.group(1))
                    if speaker in reference_files:
                        voice_config[speaker_id] = {
                            'engine': 'xtts',
                            'reference_audio': reference_files[speaker],
                            'language': target_language
                        }
                    else:
                        voice_config[speaker_id] = {
                            'engine': 'edge_tts',
                            'gender': speaker_config.get(str(speaker_id), "female")
                        }
        else:
            for speaker in unique_speakers:
                if match := re.search(r'SPEAKER_(\d+)', speaker):
                    speaker_id = int(match.group(1))
                    voice_config[speaker_id] = speaker_config.get(str(speaker_id), "female")
        
        # Step 7: Generate speech
        progress(0.7, desc="Generating speech")
        processing_status[session_id] = {"status": "Generating speech", "progress": 0.7}
        
        dubbed_audio_path = generate_tts(translated_segments, target_language, voice_config, output_dir="audio2")
        
        # Step 8: Create final video
        progress(0.85, desc="Creating final video")
        processing_status[session_id] = {"status": "Creating final video", "progress": 0.85}
        
        output_video_path = create_video_with_mixed_audio(video_path, bg_audio_path, dubbed_audio_path)
        
        progress(1.0, desc="Process completed")
        processing_status[session_id] = {"status": "Completed", "progress": 1.0}
        
        return {
            "video": output_video_path,
            "subtitle": subtitle_file,
            "message": "Process completed successfully!"
        }
        
    except Exception as e:
        logger.exception("Error in processing pipeline")
        processing_status[session_id] = {"status": f"Error: {str(e)}", "progress": -1}
        return {
            "video": None,
            "subtitle": None,
            "message": f"Error: {str(e)}"
        }

def get_processing_status(session_id):
    """Get the current processing status"""
    return processing_status.get(session_id, {}).get("status", "No status available")

def check_api_tokens():
    """Check required API tokens"""
    missing = []
    if not os.getenv("HUGGINGFACE_TOKEN"):
        missing.append("HUGGINGFACE_TOKEN")
    return f"Missing: {', '.join(missing)}" if missing else "All tokens set"

def create_interface():
    with gr.Blocks(title="SyncDub - Video Translation and Dubbing") as app:
        gr.Markdown("# SyncDub - Video Translation and Dubbing")
        
        # API status check
        api_status = check_api_tokens()
        if "Missing" in api_status:
            gr.Markdown(f"⚠️ **{api_status}**", elem_classes=["warning"])
        
        session_id = gr.State(create_session_id)
        
        with gr.Tab("Process Video"):
            with gr.Row():
                with gr.Column(scale=2):
                    media_input = gr.Textbox(label="Video Source", placeholder="URL or file path")
                    target_language = gr.Dropdown(
                        choices=["en", "es", "fr", "de", "it", "ja", "ko", "pt", "ru", "zh"],
                        label="Target Language",
                        value="en"
                    )
                    tts_choice = gr.Radio(
                        choices=["Simple dubbing (Edge TTS)", "Voice cloning (XTTS)"],
                        label="TTS Method",
                        value="Simple dubbing (Edge TTS)"
                    )
                    max_speakers = gr.Number(
                        label="Max Speakers (0 for auto)",
                        value=0,
                        precision=0,
                        minimum=0,
                        maximum=8
                    )
                    
                    # Dynamic speaker inputs
                    speaker_inputs = gr.Column()
                    
                    # Status components
                    status_text = gr.Textbox(label="Status", interactive=False)
                    process_btn = gr.Button("Process Video", variant="primary")
                
                with gr.Column(scale=3):
                    output_video = gr.Video(label="Output Video")
                    subtitle_output = gr.File(label="Subtitles")
                    output_message = gr.Textbox(label="Message")
            
            def update_speaker_inputs(tts_choice, max_speakers):
                inputs = []
                max_spk = int(max_speakers) if max_speakers > 0 else 0
                
                for i in range(max_spk):
                    if tts_choice == "Simple dubbing (Edge TTS)":
                        inputs.append(
                            gr.Radio(
                                choices=["male", "female"],
                                label=f"Speaker {i+1} Gender",
                                value="female",
                                visible=True
                            )
                        )
                    else:
                        inputs.append(
                            gr.File(
                                label=f"Speaker {i+1} Reference Audio",
                                file_types=["audio"],
                                visible=True
                            )
                        )
                
                # Add dummy hidden components for remaining slots
                for i in range(max_spk, 8):
                    inputs.append(gr.Textbox(visible=False))
                
                return inputs
            
            tts_choice.change(
                update_speaker_inputs,
                [tts_choice, max_speakers],
                [speaker_inputs]
            )
            
            max_speakers.change(
                update_speaker_inputs,
                [tts_choice, max_speakers],
                [speaker_inputs]
            )
            
            def process_wrapper(media_source, target_lang, tts_method, max_spk, *spk_inputs):
                speaker_config = {}
                for i, input_val in enumerate(spk_inputs):
                    if isinstance(input_val, dict):  # File input
                        speaker_config[str(i)] = input_val["name"]
                    elif input_val in ["male", "female"]:
                        speaker_config[str(i)] = input_val
                return process_video(
                    media_source, target_lang, tts_method, max_spk,
                    speaker_config, session_id.value
                )
            
            process_btn.click(
                process_wrapper,
                [media_input, target_language, tts_choice, max_speakers, speaker_inputs],
                [output_video, subtitle_output, output_message]
            )
            
            # Status updates
            status_timer = gr.Timer(
                interval=1000,
                fn=lambda: get_processing_status(session_id.value),
                outputs=status_text
            )
        
        with gr.Tab("Help"):
            gr.Markdown("""
            ## Usage Guide
            1. **Input**: Provide a video URL or upload a file
            2. **Language**: Select target language
            3. **TTS Method**:
               - Edge TTS: Select speaker genders
               - XTTS: Upload reference audio clips
            4. **Max Speakers**: Set maximum speakers to detect (0 for auto)
            5. **Process**: Click the Process button
            """)
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)