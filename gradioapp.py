import os
import sys
import logging
import re
import gradio as gr
from dotenv import load_dotenv

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create necessary directories
def create_directories(dirs):
    """Create necessary directories"""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

create_directories(["temp", "audio", "audio2", "reference_audio", "output"])

# Load environment variables
load_dotenv()

# Get API tokens
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    logger.error("Error: HUGGINGFACE_TOKEN not found in .env file")

# Language options
LANGUAGE_OPTIONS = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Hindi": "hi",
    "Arabic": "ar"
}

# Global variable to store speaker information for dynamic UI generation
speaker_info = {}

def process_video(video_input, youtube_url, target_language, tts_choice, max_speakers, progress=gr.Progress()):
    """Process video through the SyncDub pipeline and return intermediate outputs for UI"""
    # Initialize progress and status
    progress(0, desc="Initializing")
    status_updates = []
    
    def update_status(message):
        status_updates.append(message)
        return "\n".join(status_updates)
    
    try:
        # Initialize components
        update_status("Initializing pipeline components...")
        ingester = MediaIngester(output_dir="temp")
        recognizer = SpeechRecognizer(model_size="base")
        diarizer = SpeakerDiarizer(hf_token=hf_token)
        
        # Determine input source
        media_source = youtube_url if youtube_url else video_input
        if not media_source:
            return None, "Error: No input provided. Please upload a video or enter a YouTube URL."
        
        # Step 1: Process input and extract audio
        update_status("Processing media source...")
        progress(0.1, desc="Processing media")
        video_path = ingester.process_input(media_source)
        audio_path = ingester.extract_audio(video_path)
        
        update_status("Separating audio sources...")
        progress(0.2, desc="Separating audio")
        clean_audio_path, bg_audio_path = ingester.separate_audio_sources(audio_path)
        update_status(f"Audio processing completed. Extracted clean speech and background audio.")
        
        # Step 2: Perform speech recognition
        update_status("Transcribing audio...")
        progress(0.3, desc="Transcribing audio")
        segments = recognizer.transcribe(clean_audio_path)
        
        # Step 3: Perform speaker diarization
        update_status("Identifying speakers...")
        progress(0.4, desc="Identifying speakers")
        max_speakers_value = None if not max_speakers else int(max_speakers)
        speakers = diarizer.diarize(clean_audio_path, max_speakers=max_speakers_value)
        
        # Step 4: Assign speakers to segments
        update_status("Assigning speakers to segments...")
        progress(0.5, desc="Assigning speakers")
        final_segments = diarizer.assign_speakers_to_segments(segments, speakers)
        
        # Gather unique speakers for the next UI step
        unique_speakers = set()
        for segment in final_segments:
            if 'speaker' in segment:
                unique_speakers.add(segment['speaker'])
        
        # Extract reference audio for voice cloning if needed
        if tts_choice == "Voice cloning (XTTS)":
            update_status(f"Extracting reference audio for {len(unique_speakers)} speakers...")
            progress(0.6, desc="Extracting reference audio")
            reference_files = diarizer.extract_speaker_references(
                clean_audio_path, 
                speakers, 
                output_dir="reference_audio"
            )
        else:
            reference_files = {}
        
        # Step 5: Translate the segments
        update_status(f"Translating to {target_language}...")
        progress(0.7, desc="Translating")
        translated_segments = translate_text(
            final_segments, 
            target_lang=target_language,
            translation_method="batch"
        )
        
        # Create subtitles for preview
        subtitle_file = f"temp/{os.path.basename(video_path).split('.')[0]}_{target_language}.srt"
        generate_srt_subtitles(translated_segments, output_file=subtitle_file)
        update_status(f"Generated subtitle file: {subtitle_file}")
        
        # Prepare speaker info for the next UI step
        global speaker_info
        speaker_info = {
            "video_path": video_path,
            "bg_audio_path": bg_audio_path,
            "translated_segments": translated_segments,
            "target_language": target_language,
            "use_voice_cloning": tts_choice == "Voice cloning (XTTS)",
            "reference_files": reference_files,
            "unique_speakers": sorted(list(unique_speakers)),
            "subtitle_file": subtitle_file
        }
        
        # Get sample transcript for preview
        sample_transcript = ""
        for i, segment in enumerate(translated_segments[:5]):  # Show first 5 segments
            sample_transcript += f"Speaker {segment.get('speaker', 'Unknown')}: {segment.get('text', '')}\n"
            if i >= 4 and len(translated_segments) > 5:
                sample_transcript += f"\n... and {len(translated_segments) - 5} more segments"
        
        progress(1.0, desc="Initial processing complete")
        return sample_transcript, update_status("Initial processing complete. Please configure speaker voices in the next tab.")
    
    except Exception as e:
        logger.exception("Error processing video")
        return None, update_status(f"Error: {str(e)}")

def create_speaker_ui(speakers_detected):
    """Dynamically create UI elements for speaker configuration based on detected speakers"""
    global speaker_info
    use_voice_cloning = speaker_info.get("use_voice_cloning", False)
    reference_files = speaker_info.get("reference_files", {})
    unique_speakers = speaker_info.get("unique_speakers", [])
    
    if not unique_speakers:
        return [gr.Markdown("No speakers detected.")]
    
    ui_elements = [gr.Markdown(f"### Configure voices for {len(unique_speakers)} detected speakers")]
    speaker_configs = {}
    
    for speaker in unique_speakers:
        match = re.search(r'SPEAKER_(\d+)', speaker)
        if not match:
            continue
            
        speaker_id = int(match.group(1))
        speaker_label = f"Speaker {speaker_id+1}"
        
        # Create speaker section
        ui_elements.append(gr.Markdown(f"#### {speaker_label}"))
        
        if use_voice_cloning and speaker in reference_files:
            # Reference audio exists for this speaker
            ref_audio = reference_files[speaker]
            ui_elements.append(gr.Audio(value=ref_audio, label="Reference Audio Sample"))
            ui_elements.append(gr.Dropdown(
                choices=["Use voice cloning", "Use Edge TTS instead"], 
                value="Use voice cloning",
                label=f"Voice Option for {speaker_label}"
            ))
            gender_visible = False
            gender_value = "male"  # Default, won't be used unless Edge TTS is selected
        else:
            # No reference audio or not using voice cloning
            if use_voice_cloning:
                ui_elements.append(gr.Markdown("*No reference audio available for this speaker*"))
            
            ui_elements.append(gr.Radio(
                choices=["Male", "Female"], 
                value="Male",
                label=f"Voice Gender for {speaker_label}"
            ))
            gender_visible = True
            gender_value = "Male"
        
        # Store configuration (will be used in final processing)
        speaker_configs[speaker_id] = {
            "speaker": speaker,
            "use_cloning": use_voice_cloning and speaker in reference_files,
            "ref_audio": reference_files.get(speaker, None),
            "gender": gender_value,
            "gender_visible": gender_visible
        }
    
    speaker_info["speaker_configs"] = speaker_configs
    
    # Add final process button
    ui_elements.append(gr.Button("Generate Dubbed Video"))
    
    return ui_elements

def finalize_video(progress=gr.Progress()):
    """Generate final dubbed video with configured voices"""
    global speaker_info
    progress(0, desc="Starting final processing")
    status_updates = []
    
    def update_status(message):
        status_updates.append(message)
        return "\n".join(status_updates)
    
    try:
        # Extract needed info from global state
        video_path = speaker_info["video_path"]
        bg_audio_path = speaker_info["bg_audio_path"]
        translated_segments = speaker_info["translated_segments"]
        target_language = speaker_info["target_language"]
        use_voice_cloning = speaker_info["use_voice_cloning"]
        speaker_configs = speaker_info.get("speaker_configs", {})
        
        # Create voice configuration based on UI selections
        update_status("Configuring voices for speakers...")
        progress(0.1, desc="Configuring voices")
        
        voice_config = {}
        for speaker_id, config in speaker_configs.items():
            if use_voice_cloning and config["use_cloning"] and config["ref_audio"]:
                # Set up XTTS voice cloning
                voice_config[speaker_id] = {
                    'engine': 'xtts',
                    'reference_audio': config["ref_audio"],
                    'language': target_language
                }
                update_status(f"Using voice cloning for Speaker {speaker_id+1}")
            else:
                # Set up Edge TTS
                gender = "female" if config.get("gender", "").lower().startswith("f") else "male"
                voice_config[speaker_id] = {
                    'engine': 'edge_tts',
                    'gender': gender
                }
                update_status(f"Using Edge TTS ({gender}) for Speaker {speaker_id+1}")
        
        # Generate speech in target language
        update_status("Generating speech audio...")
        progress(0.3, desc="Generating speech")
        dubbed_audio_path = generate_tts(translated_segments, target_language, voice_config, output_dir="audio2")
        
        # Create video with mixed audio
        update_status("Creating final video with translated audio...")
        progress(0.7, desc="Creating video")
        output_video_path = create_video_with_mixed_audio(video_path, bg_audio_path, dubbed_audio_path)
        
        progress(1.0, desc="Complete")
        return output_video_path, speaker_info["subtitle_file"], update_status("Video dubbing completed successfully!")
    
    except Exception as e:
        logger.exception("Error generating dubbed video")
        return None, None, update_status(f"Error: {str(e)}")

# Create the Gradio interface
with gr.Blocks(title="SyncDub - AI Video Dubbing") as app:
    gr.Markdown("# SyncDub - AI Video Dubbing")
    gr.Markdown("Translate and dub videos with AI-powered voice cloning")
    
    with gr.Tabs() as tabs:
        with gr.Tab("1. Upload & Process"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: Choose your video source")
                    video_input = gr.Video(label="Upload Video File")
                    youtube_url = gr.Textbox(label="Or Enter YouTube URL")
                    
                    gr.Markdown("### Step 2: Choose target language and settings")
                    target_language = gr.Dropdown(
                        choices=list(LANGUAGE_OPTIONS.keys()),
                        label="Target Language",
                        value="English"
                    )
                    tts_choice = gr.Radio(
                        choices=["Simple dubbing (Edge TTS)", "Voice cloning (XTTS)"],
                        label="Voice Generation Method",
                        value="Simple dubbing (Edge TTS)"
                    )
                    max_speakers = gr.Number(
                        label="Maximum Number of Speakers (Optional)",
                        value=None,
                        precision=0
                    )
                    process_btn = gr.Button("Process Video")
                
                with gr.Column():
                    transcript_output = gr.Textbox(label="Translation Preview", lines=10)
                    status_output = gr.Textbox(label="Status", lines=10)
        
        with gr.Tab("2. Configure Voices"):
            speaker_ui_container = gr.Column()
        
        with gr.Tab("3. Final Output"):
            with gr.Row():
                with gr.Column():
                    final_status = gr.Textbox(label="Generation Status", lines=10)
                with gr.Column():
                    output_video = gr.Video(label="Dubbed Video")
                    subtitle_download = gr.File(label="Download Subtitles (SRT)")
    
    # Connect events
    process_btn.click(
        fn=lambda lang, *args: LANGUAGE_OPTIONS[lang],
        inputs=[target_language],
        outputs=[],
        _js="(lang) => {window.selected_language = lang; return lang;}"
    ).then(
        fn=process_video,
        inputs=[video_input, youtube_url, target_language, tts_choice, max_speakers],
        outputs=[transcript_output, status_output]
    ).success(
        fn=lambda: gr.Tab.update(selected=True),
        inputs=[],
        outputs=tabs[1]  # Select second tab
    ).then(
        fn=create_speaker_ui,
        inputs=[transcript_output],  # Just a dummy input
        outputs=[speaker_ui_container]
    )
    
    # Add button to speaker UI container (created dynamically)
    # The actual button will be added in create_speaker_ui function
    speaker_ui_container.children[-1].click(
        fn=finalize_video,
        inputs=[],
        outputs=[output_video, subtitle_download, final_status]
    ).success(
        fn=lambda: gr.Tab.update(selected=True),
        inputs=[],
        outputs=tabs[2]  # Select third tab
    )

if __name__ == "__main__":
    # Check if HF token is available
    if not hf_token:
        print("Warning: HUGGINGFACE_TOKEN not found in .env file")
        print("Speaker diarization may not work properly")
    
    # Launch the Gradio app
    app.launch(share=True)