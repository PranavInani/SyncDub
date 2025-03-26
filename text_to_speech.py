import numpy as np
import os
import re
import tempfile
import logging
import torch
from pydub import AudioSegment
from pathlib import Path
import subprocess
import torchaudio


# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directory structure
def ensure_directories():
    """Ensure the required directories exist"""
    directories = ["audio", "audio2", "reference_audio"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
ensure_directories()  # Call immediately to ensure directories exist

# Setup audio effects for pydub
def setup_audio_effects():
    """Setup custom audio effects"""
    from pydub import effects
    
    # Add speedup if it's missing
    if not hasattr(AudioSegment, "speedup"):
        def speedup(audio_segment, playback_speed=1.5):
            if playback_speed <= 0 or playback_speed == 1.0:
                return audio_segment
            new_frame_rate = int(audio_segment.frame_rate * playback_speed)
            adjusted = audio_segment._spawn(audio_segment.raw_data, 
                                          overrides={'frame_rate': new_frame_rate})
            return adjusted.set_frame_rate(audio_segment.frame_rate)
        AudioSegment.speedup = speedup
    
    # Add time_stretch if it's missing
    if not hasattr(effects, "time_stretch"):
        def time_stretch(audio_segment, stretch_factor):
            if stretch_factor <= 0 or stretch_factor == 1.0:
                return audio_segment
            original_frame_rate = audio_segment.frame_rate
            new_frame_rate = int(original_frame_rate / stretch_factor)
            stretched = audio_segment._spawn(
                audio_segment.raw_data,
                overrides={'frame_rate': new_frame_rate}
            )
            return stretched.set_frame_rate(original_frame_rate)
        effects.time_stretch = time_stretch
    
    return effects

effects = setup_audio_effects()

def adjust_audio_duration(audio_segment, target_duration):
    """Adjust audio to target duration by adding silence or trimming"""
    current_duration = len(audio_segment) / 1000  # ms to seconds
    
    if current_duration < target_duration:
        silence_duration_ms = int((target_duration - current_duration) * 1000)
        silence = AudioSegment.silent(duration=silence_duration_ms)
        return audio_segment + silence
    else:
        return audio_segment[:int(target_duration * 1000)]

# XTTS Model Loader (Singleton pattern)
class XTTSModelLoader:
    _instance = None
    model = None
    
    @classmethod
    def get_model(cls):
        """Get or initialize the XTTS model"""
        if cls.model is None:
            try:
                # Determine device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading XTTS model on {device}...")
                
                # Using the TTS API for convenience in loading the model
                from TTS.api import TTS
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import Xtts
                
                # Get the model info first using the API
                api_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                model_path = api_model.model_path
                config_path = os.path.join(os.path.dirname(model_path), "config.json")
                vocab_path = os.path.join(os.path.dirname(model_path), "vocab.json")
                
                # Load the model directly like in the sample code
                config = XttsConfig()
                config.load_json(config_path)
                cls.model = Xtts.init_from_config(config)
                cls.model.load_checkpoint(config, checkpoint_path=model_path, vocab_path=vocab_path, use_deepspeed=False)
                
                if device == "cuda":
                    cls.model.cuda()
                
                logger.info("XTTS model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading XTTS model: {e}")
                return None
                
        return cls.model

def create_segmented_edge_tts(text, pitch, voice, output_path, target_duration=None):
    """Create voice clone with specific characteristics and timing using Edge TTS"""
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    temp_filename = temp_file.name  # Store filename before closing
    temp_file.close()
    
    # Fix pitch formatting
    pitch_param = f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz"
    
    command = [
        "edge-tts",
        f"--pitch={pitch_param}",
        "--voice", voice,
        "--text", text,
        "--write-media", temp_filename
    ]
    subprocess.run(command, check=True)
    # Load audio
    audio = AudioSegment.from_file(temp_filename, format="mp3")
    
    # Time constraint adjustment
    if target_duration is not None:
        current_duration = len(audio) / 1000  # ms to seconds
        
        if abs(current_duration - target_duration) > 0.1:  # 100ms threshold
            speed_factor = current_duration / target_duration
            speed_factor = min(max(speed_factor, 0.7), 2.0)  # Keep within bounds
            
            logger.info(f"  Adjusting timing: {current_duration:.2f}s → {target_duration:.2f}s (factor: {speed_factor:.2f})")
            
            # Apply time adjustment
            # Instead of speed adjustments after generation, use Edge TTS rate parameter
            if speed_factor < 1:
                rate_adjustment = f"-{int((1 - speed_factor) * 100)}%"
            else:
                rate_adjustment = f"+{int((speed_factor - 1) * 100)}%"
            
            # Regenerate with adjusted rate
            os.unlink(temp_file.name)  # Remove the previous temp file
            
            # Create new command with rate parameter and fixed pitch formatting
            command = [
                "edge-tts",
                f"--pitch={pitch_param}",
                f"--rate={rate_adjustment}",
                "--voice", voice,
                "--text", text,
                "--write-media", temp_filename
            ]
            subprocess.run(command, check=True)
            
            # Reload audio with rate adjustment
            audio = AudioSegment.from_file(temp_filename, format="mp3")
            
            # Fine-tune if needed
            new_duration = len(audio) / 1000
            if abs(new_duration - target_duration) > 0.1:
                audio = adjust_audio_duration(audio, target_duration)
    
    # Save the modified audio
    audio.export(output_path, format="wav")
    
    # Clean up temporary file
    os.unlink(temp_file.name)
    
    # Log final duration
    final_audio = AudioSegment.from_file(output_path)
    final_duration = len(final_audio) / 1000
    logger.info(f"  Final duration: {final_duration:.2f}s (target: {target_duration if target_duration else 'None'}s)")
    
    return output_path

def create_segmented_xtts(text, reference_audio, language, output_path, target_duration=None):
    """Create voice-cloned speech using XTTS with speaker's reference audio"""
    # Get the model (will be loaded on first call)
    tts_model = XTTSModelLoader.get_model()
    
    if tts_model is None:
        raise RuntimeError("XTTS model could not be loaded. Ensure TTS is installed.")
    
    # Verify reference audio exists
    if not os.path.exists(reference_audio):
        raise FileNotFoundError(f"Reference audio file not found: {reference_audio}")
    
    # Default speed factor is 1.0 (no adjustment)
    speed_factor = 1.0
    
    # Get conditioning latents from reference audio
    logger.info(f"Extracting voice characteristics from reference: {os.path.basename(reference_audio)}")
    gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
        audio_path=reference_audio, 
        gpt_cond_len=tts_model.config.gpt_cond_len,
        max_ref_length=tts_model.config.max_ref_len, 
        sound_norm_refs=tts_model.config.sound_norm_refs
    )
    
    # Calculate speed adjustment if target_duration is provided
    if target_duration is not None:
        # Generate initial audio (without saving) to check duration
        logger.info(f"Estimating speech duration...")
        initial_output = tts_model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=tts_model.config.temperature,
            length_penalty=tts_model.config.length_penalty,
            repetition_penalty=tts_model.config.repetition_penalty,
            top_k=tts_model.config.top_k,
            top_p=tts_model.config.top_p,
        )
        
        # Calculate the duration of generated speech
        estimated_duration = len(initial_output["wav"]) / 24000  # XTTS uses 24kHz sample rate
        
        if abs(estimated_duration - target_duration) > 0.1:  # 100ms threshold
            # Calculate speed factor - inverse of duration ratio
            speed_factor = estimated_duration / target_duration
            speed_factor = min(max(speed_factor, 0.7), 1.5)  # Keep in reasonable range
            logger.info(f"  Adjusting timing: {estimated_duration:.2f}s → {target_duration:.2f}s (speed factor: {speed_factor:.2f})")
    
    # Generate speech with the proper speed factor using the direct speed parameter
    logger.info(f"Generating XTTS speech using reference: {os.path.basename(reference_audio)}")
    
    # Run inference with adjusted parameters
    output = tts_model.inference(
        text=text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=tts_model.config.temperature,
        length_penalty=tts_model.config.length_penalty,
        repetition_penalty=tts_model.config.repetition_penalty,
        top_k=tts_model.config.top_k,
        top_p=tts_model.config.top_p,
        speed=speed_factor,  # Use the direct speed parameter
    )
    
    # Convert the output to proper audio format and save
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_filename = temp_file.name
    temp_file.close()
    
    # Save using torchaudio for better compatibility
    wav_tensor = torch.tensor(output["wav"]).unsqueeze(0)
    torchaudio.save(temp_filename, wav_tensor, 24000)  # XTTS uses 24kHz sample rate
    
    # Load the audio file with pydub for further processing
    audio = AudioSegment.from_file(temp_filename)
    
    # Fine tune if needed - keep this as a fallback since extreme speed values can cause artifacts
    if target_duration is not None:
        current_duration = len(audio) / 1000  # ms to seconds
        if abs(current_duration - target_duration) > 0.1:
            logger.info(f"  Fine-tuning duration with post-processing: {current_duration:.2f}s → {target_duration:.2f}s")
            audio = adjust_audio_duration(audio, target_duration)
    
    # Save the final audio
    audio.export(output_path, format="wav")
    
    # Clean up
    os.unlink(temp_filename)
    
    # Log final duration
    final_audio = AudioSegment.from_file(output_path)
    final_duration = len(final_audio) / 1000
    logger.info(f"  Final duration: {final_duration:.2f}s (target: {target_duration if target_duration else 'None'}s)")
    
    return output_path

def process_voice_config(voice_config):
    """
    Process voice configuration to support both Edge TTS and XTTS
    
    Args:
        voice_config: Dict with speaker_id keys and configuration values
            For Edge TTS: {'engine': 'edge_tts', 'gender': 'male'/'female'} or simply 'male'/'female'
            For XTTS: {'engine': 'xtts', 'reference_audio': '/path/to/audio.wav', 'language': 'hi'}
    
    Returns:
        Processed configuration dictionary
    """
    processed_config = {}
    
    # Handle empty config
    if not voice_config:
        return {0: {'engine': 'edge_tts', 'voice': "hi-IN-MadhurNeural", 'pitch': 0}}
    
    # Track Edge TTS speaker counts for pitch variations
    edge_male_count = 0
    edge_female_count = 0
    
    # Pitch variations for multiple Edge TTS speakers of same gender
    male_pitches = [0, -30, 40]  # Default, deeper, higher
    female_pitches = [0, 25, -25]  # Default, higher, deeper
    
    for speaker_id, config in voice_config.items():
        # Convert string speaker_id to int if needed
        if isinstance(speaker_id, str) and speaker_id.isdigit():
            speaker_id = int(speaker_id)
        
        # Determine which engine to use (default is edge_tts)
        if isinstance(config, dict):
            engine = config.get('engine', 'edge_tts')
        else:
            # Handle simple gender strings for backwards compatibility
            engine = 'edge_tts'
            config = {'gender': config} if config in ['male', 'female'] else {'gender': 'male'}
        
        if engine == 'xtts':
            # XTTS configuration - each speaker needs their own reference audio
            if 'reference_audio' not in config:
                logger.warning(f"No reference audio provided for XTTS speaker {speaker_id}, falling back to Edge TTS")
                # Fall back to Edge TTS if no reference audio
                engine = 'edge_tts'
                gender = config.get('gender', 'male')
            else:
                # Valid XTTS configuration
                processed_config[speaker_id] = {
                    'engine': 'xtts',
                    'reference_audio': config['reference_audio'],
                    'language': config.get('language', 'hi')  # Default to Hindi
                }
                continue  # Skip the Edge TTS processing below
        
        # Edge TTS configuration (if engine is edge_tts or XTTS fallback)
        gender = config.get('gender', 'male')
        
        if gender == 'male':
            # Assign male voice and pitch
            pitch = male_pitches[edge_male_count % len(male_pitches)]
            processed_config[speaker_id] = {
                'engine': 'edge_tts',
                'voice': "hi-IN-MadhurNeural",
                'pitch': pitch
            }
            edge_male_count += 1
        else:
            # Assign female voice and pitch
            pitch = female_pitches[edge_female_count % len(female_pitches)]
            processed_config[speaker_id] = {
                'engine': 'edge_tts',
                'voice': "hi-IN-SwaraNeural", 
                'pitch': pitch
            }
            edge_female_count += 1
    
    return processed_config

def generate_tts(segments, target_language, voice_config=None, output_dir="audio2"):
    """
    Generate speech for all segments using appropriate TTS engine per speaker
    
    Args:
        segments: List of segments with text, speaker, start and end times
        target_language: Language code for TTS
        voice_config: Dictionary with speaker configurations
                     - For Edge TTS: {'gender': 'male'/'female'} or just 'male'/'female'
                     - For XTTS: {'engine': 'xtts', 'reference_audio': '/path/to/audio.wav'}
        output_dir: Directory to save the final audio
        
    Returns:
        Path to the final combined audio file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the full audio
    output_path = os.path.join(output_dir, "dubbed_conversation.wav")
    max_end_time = max(segment['end'] for segment in segments)
    
    # Create a silent audio of the total duration
    combined = AudioSegment.silent(duration=int(max_end_time * 1000) + 100) 
    ensure_directories()
    audio_files = []
    
    # Process voice configuration
    processed_config = process_voice_config(voice_config or {})
    print(processed_config)
    
    # Process each segment
    for i, segment in enumerate(segments):
        # Extract speaker ID
        speaker = segment.get('speaker', 'SPEAKER_00')
        match = re.search(r'SPEAKER_(\d+)', speaker)
        speaker_id = int(match.group(1)) if match else 0
        
        # Get speaker configuration
        speaker_config = processed_config.get(speaker_id, 
                                             {'engine': 'edge_tts', 'voice': "hi-IN-SwaraNeural", 'pitch': 0})
        
        # Get text and timing information
        text = segment['text']
        start = segment['start']
        end = segment['end']
        duration = end - start
        
        # Create output filename
        output_file = f"audio/{start}.wav"
        
        logger.info(f"Processing segment {i+1} (Speaker {speaker_id}, Engine: {speaker_config['engine']}):")
        logger.info(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        logger.info(f"  Duration: {duration:.2f}s")
        
        # Choose appropriate TTS engine
        if speaker_config['engine'] == 'xtts':
            # XTTS generation with speaker's reference audio
            try:
                create_segmented_xtts(
                    text=text,
                    reference_audio=speaker_config['reference_audio'],
                    language=speaker_config.get('language', target_language),
                    output_path=output_file,
                    target_duration=duration,
                )
            except Exception as e:
                logger.error(f"Error using XTTS for speaker {speaker_id}: {e}")
                logger.warning(f"Falling back to Edge TTS for this segment")
                # Fallback to Edge TTS
                create_segmented_edge_tts(
                    text=text,
                    pitch=0, 
                    voice="hi-IN-SwaraNeural",
                    output_path=output_file,
                    target_duration=duration,
                )
        else:
            # Edge TTS generation
            create_segmented_edge_tts(
                text=text,
                pitch=speaker_config.get('pitch', 0),
                voice=speaker_config.get('voice', "hi-IN-SwaraNeural"),
                output_path=output_file,
                target_duration=duration,
            )
        
        audio_files.append(output_file)

        # Add segment to combined audio at the exact timestamp
        segment_audio = AudioSegment.from_file(output_file)
        position_ms = int(segment['start'] * 1000)
        combined = combined.overlay(segment_audio, position=position_ms)
    
    # Export the final combined audio
    combined.export(output_path, format="wav")
    logger.info(f"  Final combined duration: {len(combined) / 1000:.2f}s")
    
    # Clean up segment files
    for file in audio_files:
        try:
            os.remove(file)
        except:
            pass
    
    # Verify the final duration
    final_audio = AudioSegment.from_file(output_path)
    final_duration_sec = len(final_audio) / 1000
    
    print(f"\nTarget duration: {max_end_time:.2f} seconds")
    print(f"Actual duration: {final_duration_sec:.2f} seconds")
    
    # If the final audio is still too long, trim it
    if final_duration_sec > max_end_time + 0.1:  # Allow 100ms grace
        trimmed = final_audio[:int(max_end_time * 1000)]
        trimmed.export(output_path, format="wav")
        print(f"Trimmed to exactly {max_end_time:.2f} seconds")
    
    return output_path



