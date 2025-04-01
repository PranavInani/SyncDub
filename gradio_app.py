import os
import sys
import logging
import tempfile
import re
import gradio as gr
from dotenv import load_dotenv
import threading
import shutil

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
os.makedirs("outputs", exist_ok=True)  # Add directory for downloadable outputs

# Global variables for process tracking
processing_status = {}

def create_session_id():
    """Create a unique session ID for tracking progress"""
    import uuid
    return str(uuid.uuid4())[:8]

def process_video(media_source, target_language, tts_choice, max_speakers, speaker_genders, session_id, translation_method="batch", progress=gr.Progress()):
    """Main processing function that handles the complete pipeline"""
    global processing_status
    processing_status[session_id] = {"status": "Starting", "progress": 0}
    
    try:
        # Get API tokens
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        if not hf_token:
            return {"error": True, "message": "Error: HUGGINGFACE_TOKEN not found in .env file"}
        
        # Determine if input is URL or file
        is_url = media_source.startswith(("http://", "https://"))
        
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
        
        # Convert max_speakers to int or None
        max_speakers_val = int(max_speakers) if max_speakers and max_speakers.strip() else None
        
        # Diarize audio to identify speakers
        speakers = diarizer.diarize(clean_audio_path, max_speakers=max_speakers_val)
        
        # Step 4: Assign speakers to segments
        progress(0.4, desc="Assigning speakers to segments")
        processing_status[session_id] = {"status": "Assigning speakers to segments", "progress": 0.4}
        
        final_segments = diarizer.assign_speakers_to_segments(segments, speakers)
        
        # Step 5: Translate the segments
        progress(0.5, desc=f"Translating to {target_language}")
        processing_status[session_id] = {"status": f"Translating to {target_language}", "progress": 0.5}
        
        # Validate target language
        valid_languages = ["en", "es", "fr", "de", "it", "ja", "ko", "pt", "ru", "zh", "hi"]
        if target_language not in valid_languages:
            logger.warning(f"Unsupported language: {target_language}, falling back to English")
            target_language = "en"
        
        translated_segments = translate_text(
            final_segments, 
            target_lang=target_language,
            translation_method=translation_method
        )
        
        # Generate subtitle file
        subtitle_file = f"temp/{os.path.basename(video_path).split('.')[0]}_{target_language}.srt"
        generate_srt_subtitles(translated_segments, output_file=subtitle_file)
        
        # Step 6: Configure voice characteristics for speakers
        progress(0.6, desc="Configuring voices")
        processing_status[session_id] = {"status": "Configuring voices", "progress": 0.6}
        
        # Detect number of unique speakers
        unique_speakers = set()
        for segment in translated_segments:
            if 'speaker' in segment:
                unique_speakers.add(segment['speaker'])
        
        logger.info(f"Detected {len(unique_speakers)} speakers")
        
        # Use provided speaker genders
        use_voice_cloning = tts_choice == "Voice cloning (XTTS)"
        voice_config = {}  # Map of speaker_id to gender or voice config
        
        if use_voice_cloning:
            # Extract reference audio for voice cloning
            logger.info("Extracting speaker reference audio for voice cloning...")
            reference_files = diarizer.extract_speaker_references(
                clean_audio_path, 
                speakers, 
                output_dir="reference_audio"
            )
            
            # Create voice config for XTTS
            for speaker in sorted(list(unique_speakers)):
                match = re.search(r'SPEAKER_(\d+)', speaker)
                if match:
                    speaker_id = int(match.group(1))
                    if speaker in reference_files:
                        voice_config[speaker_id] = {
                            'engine': 'xtts',
                            'reference_audio': reference_files[speaker],
                            'language': target_language  # Use the validated target language
                        }
                        logger.info(f"Using voice cloning for Speaker {speaker_id+1} with reference file: {os.path.basename(reference_files[speaker])}")
                    else:
                        # Fallback to Edge TTS if no reference audio
                        logger.warning(f"No reference audio found for Speaker {speaker_id+1}, falling back to Edge TTS")
                        gender = "female"  # Default fallback
                        if str(speaker_id) in speaker_genders and speaker_genders[str(speaker_id)]:
                            gender = speaker_genders[str(speaker_id)]
                        
                        voice_config[speaker_id] = {
                            'engine': 'edge_tts',
                            'gender': gender
                        }
        else:
            # Standard Edge TTS configuration
            if len(unique_speakers) > 0:
                for speaker in sorted(list(unique_speakers)):
                    match = re.search(r'SPEAKER_(\d+)', speaker)
                    if match:
                        speaker_id = int(match.group(1))
                        gender = "female" if speaker_id % 2 == 0 else "male"  # Default fallback
                        
                        # Use selected gender if available
                        if str(speaker_id) in speaker_genders and speaker_genders[str(speaker_id)]:
                            gender = speaker_genders[str(speaker_id)]
                            
                        voice_config[speaker_id] = gender
        
        # Step 7: Generate speech in target language
        progress(0.7, desc=f"Generating speech in {target_language}")
        processing_status[session_id] = {"status": f"Generating speech in {target_language}", "progress": 0.7}
        
        dubbed_audio_path = generate_tts(translated_segments, target_language, voice_config, output_dir="audio2")
        
        # Step 8: Create video with mixed audio
        progress(0.85, desc="Creating final video")
        processing_status[session_id] = {"status": "Creating final video", "progress": 0.85}
        
        success = create_video_with_mixed_audio(
            main_video_path=video_path, 
            background_music_path=bg_audio_path, 
            main_audio_path=dubbed_audio_path
        )
        
        if not success:
            raise RuntimeError("Failed to create final video with audio")
        
        # Use known output path since function returns boolean
        output_video_path = os.path.join("temp", "output_video.mp4")
        
        # Verify the output video exists
        if not os.path.exists(output_video_path):
            raise FileNotFoundError(f"Output video not found at expected path: {output_video_path}")
        
        # Create downloadable copies with unique names
        file_basename = os.path.basename(video_path).split('.')[0]
        downloadable_video = f"outputs/{file_basename}_{target_language}_{session_id}.mp4"
        downloadable_subtitle = f"outputs/{file_basename}_{target_language}_{session_id}.srt"
        
        # Copy files to outputs directory for download
        shutil.copy2(output_video_path, downloadable_video)
        shutil.copy2(subtitle_file, downloadable_subtitle)
            
        # Complete
        progress(1.0, desc="Process completed")
        processing_status[session_id] = {"status": "Completed", "progress": 1.0}
        
        return {
            "error": False,
            "video": downloadable_video,
            "subtitle": downloadable_subtitle,
            "message": "Process completed successfully! Click on the files to download."
        }
        
    except Exception as e:
        logger.exception("Error in processing pipeline")
        processing_status[session_id] = {"status": f"Error: {str(e)}", "progress": -1}
        return {"error": True, "message": f"Error: {str(e)}"}

def get_processing_status(session_id):
    """Get the current processing status for the given session"""
    global processing_status
    if session_id in processing_status:
        return processing_status[session_id]["status"]
    return "No status available"

def check_api_tokens():
    """Check if required API tokens are set"""
    missing_tokens = []
    
    if not os.getenv("HUGGINGFACE_TOKEN"):
        missing_tokens.append("HUGGINGFACE_TOKEN")
    
    if missing_tokens:
        return f"Warning: Missing API tokens: {', '.join(missing_tokens)}. Please set them in your .env file."
    else:
        return "All required API tokens are set."

# Define the Gradio interface
def create_interface():
    with gr.Blocks(title="SyncDub - Video Translation and Dubbing") as app:
        gr.Markdown("# SyncDub - Video Translation and Dubbing")
        gr.Markdown("Translate and dub videos to different languages with speaker diarization")
        
        session_id = create_session_id()  # Create a session ID for tracking progress
        
        with gr.Tab("Process Video"):
            with gr.Row():
                with gr.Column(scale=2):
                    media_input = gr.Textbox(label="Video URL or File Upload", placeholder="Enter a YouTube URL or upload a video file")
                    
                    with gr.Row():
                        # Enhanced language dropdown with full language names
                        target_language = gr.Dropdown(
                            choices=[
                                ("English", "en"), 
                                ("Spanish", "es"), 
                                ("French", "fr"), 
                                ("German", "de"), 
                                ("Hindi", "hi"),
                                ("Italian", "it"), 
                                ("Japanese", "ja"), 
                                ("Korean", "ko"), 
                                ("Portuguese", "pt"), 
                                ("Russian", "ru"), 
                                ("Chinese", "zh")
                            ],
                            label="Target Language",
                            value="hi"
                        )
                        tts_choice = gr.Radio(
                            choices=["Simple dubbing (Edge TTS)", "Voice cloning (XTTS)"],
                            label="TTS Method",
                            value="Simple dubbing (Edge TTS)"
                        )
                    
                    # Add translation method selection
                    with gr.Row():
                        translation_method = gr.Radio(
                            choices=["batch", "iterative", "groq"],
                            label="Translation Method",
                            value="batch",
                            info="Batch: Faster for longer content. Iterative: May be more accurate for short content. Groq: Uses Groq LLM API."
                        )
                    
                    # Speaker count input and update button
                    with gr.Row():
                        max_speakers = gr.Textbox(label="Maximum number of speakers", placeholder="Leave blank for auto")
                        update_speakers_btn = gr.Button("Update Speaker Options")
                    
                    # Speaker gender container
                    with gr.Group(visible=False) as speaker_genders_container:
                        gr.Markdown("### Speaker Gender Selection")
                        speaker_genders = {}
                        for i in range(8):  # Support up to 8 speakers
                            speaker_genders[str(i)] = gr.Radio(
                                choices=["male", "female"],
                                value="male" if i % 2 == 1 else "female",
                                label=f"Speaker {i} Gender",
                                visible=False  # Initially hidden
                            )
                    
                    process_btn = gr.Button("Process Video", variant="primary")
                    status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
                
                with gr.Column(scale=3):
                    # Replace video display with file downloads
                    gr.Markdown("### Output Files")
                    output_message = gr.Textbox(label="Status", interactive=False)
                    with gr.Row():
                        output = gr.File(label="Download Video")
                        subtitle_output = gr.File(label="Download Subtitles")
            
            # Function to update speaker gender options
            def update_speaker_options(max_speakers_value):
                updates = {}
                
                try:
                    num_speakers = int(max_speakers_value) if max_speakers_value.strip() else 0
                    
                    if num_speakers > 0:
                        # Show the speaker gender container
                        updates[speaker_genders_container] = gr.Group(visible=True)
                        
                        # Show only the relevant number of speaker options
                        for i in range(8):
                            updates[speaker_genders[str(i)]] = gr.Radio(
                                visible=(i < num_speakers)
                            )
                    else:
                        # Hide all if no valid number
                        updates[speaker_genders_container] = gr.Group(visible=False)
                except ValueError:
                    # Hide all if invalid number
                    updates[speaker_genders_container] = gr.Group(visible=False)
                
                return updates
            
            # Connect the update button to show/hide speaker options
            update_speakers_btn.click(
                fn=update_speaker_options,
                inputs=[max_speakers],
                outputs=[speaker_genders_container] + [speaker_genders[str(i)] for i in range(8)]
            )
            
            # Function to actually pass the gender values to the process_video function
            def process_with_genders(media_source, target_language, tts_choice, max_speakers, translation_method, *gender_values):
                # Convert the gender values into a dictionary to pass to process_video
                speaker_genders_dict = {str(i): gender for i, gender in enumerate(gender_values) if gender}
                result = process_video(media_source, target_language, tts_choice, max_speakers, 
                                      speaker_genders_dict, session_id, translation_method=translation_method)
                
                # Return the output values based on whether there was an error
                if result.get("error", False):
                    return None, None, result.get("message", "An error occurred")
                else:
                    return result.get("video"), result.get("subtitle"), result.get("message")
            
            # Connect the process button
            process_btn.click(
                fn=process_with_genders, 
                inputs=[
                    media_input, 
                    target_language, 
                    tts_choice, 
                    max_speakers,
                    translation_method,  # Add translation method to inputs
                    # Pass individual radio components, not a Group
                    *[speaker_genders[str(i)] for i in range(8)]
                ],
                outputs=[output, subtitle_output, output_message]
            )
            
            # Update status periodically
            status_timer = gr.Timer(2, lambda: get_processing_status(session_id), None, status_text)
            
            # Create a more compatible approach for status updates
            def start_status_updates(session_id):
                def update_status_thread():
                    import time
                    while session_id in processing_status and processing_status[session_id]["progress"] < 1.0:
                        try:
                            time.sleep(1)  # Update status every second
                            # This is a workaround since we can't use JavaScript directly
                        except:
                            break
                
                thread = threading.Thread(target=update_status_thread)
                thread.daemon = True  # Thread will exit when main program exits
                thread.start()
                return "Processing started"
            
            # Manual refresh button as a fallback option
            refresh_btn = gr.Button("Refresh Status")
            
            # Status checking function
            def check_status(session_id):
                status = get_processing_status(session_id)
                return status
            
            # Connect the refresh button to check status
            refresh_btn.click(
                fn=check_status,
                inputs=[gr.State(session_id)],
                outputs=[status_text]
            )
            
            # Create a simple auto-refresh component using a Textbox with a timer
            gr.HTML("""
            <script>
            // Simple poller to update status
            document.addEventListener('DOMContentLoaded', function() {
                let refreshInterval;
                
                // Look for the primary button (Process Video)
                const processButton = document.querySelector('button.primary');
                
                if (processButton) {
                    // When process starts, begin polling
                    processButton.addEventListener('click', function() {
                        if (refreshInterval) clearInterval(refreshInterval);
                        
                        // Find the refresh button
                        const refreshButtons = Array.from(document.querySelectorAll('button'));
                        const refreshButton = refreshButtons.find(btn => btn.textContent.includes('Refresh Status'));
                        
                        if (refreshButton) {
                            // Start auto-polling every 2 seconds
                            refreshInterval = setInterval(function() {
                                refreshButton.click();
                            }, 2000);
                            
                            // Stop polling after 30 minutes (safety)
                            setTimeout(function() {
                                if (refreshInterval) clearInterval(refreshInterval);
                            }, 30*60*1000);
                        }
                    });
                }
            });
            </script>
            """)
            
        with gr.Tab("Help"):
            gr.Markdown("""
            ## How to use SyncDub
            
            1. **Input**: Enter a YouTube URL or path to a local video file, or upload a video
            2. **Target Language**: Select the language you want to translate and dub into
            3. **TTS Engine**: 
               - **Simple dubbing**: Uses Edge TTS (faster but less natural sounding)
               - **Voice cloning**: Uses XTTS to clone the original speakers' voices (slower but more natural)
            4. **Maximum Speakers**: Optionally specify the maximum number of speakers to detect
            5. **Translation Method**: Choose the translation method (Batch, Iterative, or Groq)
            6. **Process**: Click the Process Video button to start
            
            ## Requirements
            
            Make sure you have the following API tokens in your `.env` file:
            - `HUGGINGFACE_TOKEN`: Required for speech diarization
            
            ## Troubleshooting
            
            - If you encounter errors, check that all API tokens are set correctly
            - For large videos, the process may take several minutes
            - If voice cloning doesn't sound right, try simple dubbing instead
            """)
    
    return app

# Launch the interface
if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)
