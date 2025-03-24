import numpy as np
import os
import re
import tempfile
import logging
from pydub import AudioSegment
from pathlib import Path
import subprocess


# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directory structure
def ensure_directories():
    """Ensure the required directories exist"""
    directories = ["audio", "audio2"]
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

def create_segmented_edge_tts(text, pitch, voice, output_path, target_duration=None):
    """Create voice clone with specific characteristics and timing"""
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    temp_filename = temp_file.name  # Store filename before closing
    temp_file.close()
    command = [
        "edge-tts",
        f"--pitch=+{pitch}Hz",
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
            rate_adjustment = f"{int((speed_factor - 1) * 100)}%"
            if speed_factor < 1:
                rate_adjustment = f"-{int((1 - speed_factor) * 100)}%"
            else:
                rate_adjustment = f"+{int((speed_factor - 1) * 100)}%"
            
            # Regenerate with adjusted rate
            os.unlink(temp_file.name)  # Remove the previous temp file
            
            # Create new command with rate parameter
            command = [
                "edge-tts",
                f"--pitch=+{pitch}Hz",
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

def configure_voice_characteristics(voice_config):
    """
    Enhances the voice_config by adding specific voice models and pitch values
    to create distinct voices for each speaker.
    
    Args:
        voice_config: A dictionary mapping speaker_id to gender ('male' or 'female')
    
    Returns:
        Enhanced voice_config with voice model and pitch values
    """
    # Default voice models for Hindi
    male_voice = "hi-IN-MadhurNeural"
    female_voice = "hi-IN-SwaraNeural"
    
    # Pitch variations for multiple speakers of same gender
    male_pitches = [0, -30, 40]  # Default, deeper, higher
    female_pitches = [0, 25, -25]  # Default, higher, deeper
    
    # Handle empty or None voice_config
    if not voice_config:
        return {0: {'voice': male_voice, 'pitch': 0}}
    
    # Track current male and female speaker indices
    current_male = 0
    current_female = 0
    
    # Enhanced voice configuration
    enhanced_config = {}
    
    for speaker_id, gender in voice_config.items():
        # Ensure speaker_id is an integer (convert if it's a string)
        if isinstance(speaker_id, str) and speaker_id.isdigit():
            speaker_id = int(speaker_id)
            
        if gender == "male":
            # Assign voice and pitch based on male index
            pitch = male_pitches[current_male % len(male_pitches)]
            enhanced_config[speaker_id] = {
                'voice': male_voice,
                'pitch': pitch
            }
            current_male += 1
        else:  # female
            # Assign voice and pitch based on female index
            pitch = female_pitches[current_female % len(female_pitches)]
            enhanced_config[speaker_id] = {
                'voice': female_voice,
                'pitch': pitch
            }
            current_female += 1
    
    return enhanced_config

def generate_edge_tts(segments, target_language, voice_config=None,output_dir="audio2"):
    """
    Generate speech for all segments
    
    Args:
        segments: List of segments with text, speaker, start and end times
        target_language: Language code for TTS
        voice_config: Dictionary mapping speaker IDs to genders ('male'/'female')
        
    Returns:
        List of created audio files
    """
    # Ensure output directory exists
    os.makedirs(f"{output_dir}", exist_ok=True)
    # Generate the full audio
    output_path = os.path.join(output_dir, "dubbed_conversation.wav")
    max_end_time = max(segment['end'] for segment in segments)
    
    # Create a silent audio of the total duration
    combined = AudioSegment.silent(duration=int(max_end_time * 1000) + 100) 
    ensure_directories()
    audio_files = []
    
    voice_config = configure_voice_characteristics(voice_config)

    # Default voice configuration if none provided
    if voice_config is None:
        voice_config = {}
    
    # Process each segment
    for i, segment in enumerate(segments):
        # Extract speaker ID
        speaker = segment.get('speaker', 'SPEAKER_00')
        match = re.search(r'SPEAKER_(\d+)', speaker)
        speaker_id = int(match.group(1)) if match else 0
                
        voice = voice_config[speaker_id].get('voice', "hi-IN-SwaraNeural")
        pitch = voice_config[speaker_id].get('pitch', 0)
        
        # Get text and timing information
        text = segment['text']
        start = segment['start']
        end = segment['end']
        duration = end - start
        
        # Create output filename
        output_file = f"audio/{start}.wav"
        
        logger.info(f"Processing segment {i+1} (Speaker {speaker_id}, Voice: {voice}):")
        logger.info(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        logger.info(f"  Duration: {duration:.2f}s")
        
        # Generate the voice
        create_segmented_edge_tts(
            text=text,
            pitch=pitch,
            voice=voice,
            output_path=output_file,
            target_duration=duration,
        )
        
        audio_files.append(output_file)

        # Add segment to combined audio at the exact timestamp
        segment_audio = AudioSegment.from_file(output_file)
        # Position in ms
        position_ms = int(segment['start'] * 1000)
        # Add to combined audio
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


