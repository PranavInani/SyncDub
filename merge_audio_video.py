import os
import logging
import subprocess
from pydub import AudioSegment
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_video_duration(video_path):
    """
    Get the duration of a video file using ffprobe
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration of the video in seconds
    """
    duration_cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        video_path
    ]
    
    try:
        video_duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
        logger.info(f"Video duration: {video_duration:.2f} seconds")
        return video_duration
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting video duration: {e}")
        return None

def combine_audio_segments(segments, max_duration=None):
    """
    Combine all audio segments into a single audio file
    
    Args:
        segments: List of segments with start times
        max_duration: Maximum duration of the output audio (optional)
        
    Returns:
        Path to the combined audio file
    """
    # Determine the maximum duration if not specified
    if max_duration is None:
        if not segments:
            logger.error("No segments provided")
            return None
        max_duration = max(segment.get("end", 0) for segment in segments)
    
    # Add a small buffer
    max_duration += 0.1
    
    # Create a silent base audio track
    base_audio = AudioSegment.silent(duration=int(max_duration * 1000))
    
    # Overlay all audio segments at their correct positions
    logger.info(f"Combining {len(segments)} audio segments...")
    segments_found = 0
    
    for segment in segments:
        start_time = segment["start"]
        
        # Audio file path
        audio_file = f"audio2/audio/{start_time}.ogg"
        
        if os.path.exists(audio_file):
            # Load the audio segment
            audio_segment = AudioSegment.from_file(audio_file)
            
            # Position in milliseconds
            position_ms = int(start_time * 1000)
            
            # Overlay at the exact position
            base_audio = base_audio.overlay(audio_segment, position=position_ms)
            segments_found += 1
        else:
            logger.warning(f"Audio segment not found: {audio_file}")
    
    logger.info(f"Found {segments_found} out of {len(segments)} audio segments")
    
    # Create temp file for the combined audio
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_audio_path = temp_audio.name
    temp_audio.close()
    
    # Export the combined audio
    base_audio.export(temp_audio_path, format="wav")
    logger.info(f"Combined audio created: {temp_audio_path}")
    
    return temp_audio_path

def merge_audio_with_video(video_path, audio_path, output_path):
    """
    Merge audio with video using ffmpeg
    
    Args:
        video_path: Path to the input video file
        audio_path: Path to the audio file
        output_path: Path for the output video file
        
    Returns:
        True if successful, False otherwise
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # FFmpeg command to merge audio and video
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v",  # Use video from first input
        "-map", "1:a",  # Use audio from second input
        "-c:v", "copy",  # Copy video codec (no re-encoding)
        "-shortest",  # End when shortest input ends
        "-y",  # Overwrite output file if it exists
        output_path
    ]
    
    logger.info(f"Merging audio with video to create: {output_path}")
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE)
        logger.info("Merge completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error merging audio with video: {e}")
        logger.error(f"FFmpeg error output: {e.stderr.decode('utf-8')}")
        return False

def create_dubbed_video(video_path, segments, output_path=None):
    """
    Create a dubbed video by merging audio segments with the original video
    
    Args:
        video_path: Path to the original video
        segments: List of segments with start times
        output_path: Path for the output video (optional)
        
    Returns:
        Path to the dubbed video if successful, None otherwise
    """
    # Determine output path if not provided
    if output_path is None:
        video_basename = os.path.basename(video_path)
        video_name = os.path.splitext(video_basename)[0]
        output_path = os.path.join("hindi_dubbing_output", f"{video_name}_dubbed.mp4")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info("Creating dubbed video...")
    logger.info(f"Input video: {video_path}")
    logger.info(f"Output path: {output_path}")
    
    # Get video duration
    video_duration = get_video_duration(video_path)
    if not video_duration:
        return None
    
    # Combine audio segments
    combined_audio = combine_audio_segments(segments, max_duration=video_duration)
    if not combined_audio:
        return None
    
    try:
        # Merge combined audio with video
        success = merge_audio_with_video(video_path, combined_audio, output_path)
        
        # Clean up the temporary audio file
        if os.path.exists(combined_audio):
            os.unlink(combined_audio)
            logger.info(f"Removed temporary audio file: {combined_audio}")
        
        if success:
            logger.info(f"Successfully created dubbed video: {output_path}")
            return output_path
        else:
            logger.error("Failed to create dubbed video")
            return None
            
    except Exception as e:
        logger.error(f"Error during video creation: {str(e)}")
        
        # Clean up the temporary audio file
        if os.path.exists(combined_audio):
            os.unlink(combined_audio)
        
        return None

if __name__ == "__main__":
    # Example standalone usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge audio segments with a video file")
    parser.add_argument("video_path", help="Path to the original video file")
    parser.add_argument("--output", "-o", help="Path for the output dubbed video")
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        exit(1)
    
    # Set default output path if not provided
    output_path = args.output
    if not output_path:
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        output_path = f"{video_name}_dubbed.mp4"
    
    # Load segment information
    # In a real scenario, this would come from the main pipeline
    # For this example, we'll scan the audio2/audio directory for .ogg files
    logger.info("Scanning for audio segments...")
    segments = []
    audio_dir = "audio2/audio"
    
    if os.path.exists(audio_dir):
        for filename in os.listdir(audio_dir):
            if filename.endswith(".ogg"):
                # Extract start time from filename
                try:
                    start_time = float(os.path.splitext(filename)[0])
                    # Estimate end time (just for example)
                    # In real usage, this would come from the actual segments data
                    audio = AudioSegment.from_file(os.path.join(audio_dir, filename))
                    duration = len(audio) / 1000  # Convert ms to seconds
                    end_time = start_time + duration
                    
                    segments.append({
                        "start": start_time,
                        "end": end_time
                    })
                except ValueError:
                    logger.warning(f"Couldn't parse start time from filename: {filename}")
    else:
        logger.error(f"Audio directory not found: {audio_dir}")
        exit(1)
    
    if not segments:
        logger.error("No audio segments found")
        exit(1)
    
    logger.info(f"Found {len(segments)} audio segments")
    
    # Create the dubbed video
    dubbed_video = create_dubbed_video(args.video_path, segments, output_path)
    
    if dubbed_video:
        logger.info(f"Dubbed video created: {dubbed_video}")
    else:
        logger.error("Failed to create dubbed video")
        exit(1)
