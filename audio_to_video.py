import os
import subprocess
from IPython.display import FileLink, display

def create_video_with_mixed_audio(
    main_video_path, 
    background_music_path, 
    main_audio_path, 
    output_path=None,  # Add output_path parameter for custom output location
    temp_dir="temp",  # Directory for temporary files
    bg_volume=0.3,
    main_audio_volume=1.0
):
    """
    Create a video with mixed audio (main audio + background music)
    
    Parameters:
        main_video_path (str): Path to the main video file
        background_music_path (str): Path to the background music file
        main_audio_path (str): Path to the main audio (dubbed speech)
        output_path (str, optional): Custom output path for the final video
        temp_dir (str): Directory for temporary files
        bg_volume (float): Volume level for background music (0.0-1.0)
        main_audio_volume (float): Volume level for main audio (0.0-1.0)
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        # Ensure the temporary directory exists
        os.makedirs(temp_dir, exist_ok=True)
        
        # Define paths for temporary and output files
        temp_audio_path = os.path.join(temp_dir, "mixed_audio.wav")
        output_video_path = output_path if output_path else os.path.join(temp_dir, "output_video.mp4")
        
        # Step 1: Mix the background audio and main audio with volume control
        print("Step 1: Mixing audio tracks...")
        mix_command = f'''ffmpeg -i "{background_music_path}" -i "{main_audio_path}" -filter_complex \
            "[0:a]volume={bg_volume}[bg]; \
             [1:a]volume={main_audio_volume}[main]; \
             [bg][main]amix=inputs=2:duration=first:dropout_transition=2" \
            "{temp_audio_path}" -y'''
        
        subprocess.run(mix_command, shell=True, check=True)
        
        # Step 2: Replace the original audio in the video with mixed audio
        print("Step 2: Creating final video with mixed audio...")
        video_command = f'''ffmpeg -i "{main_video_path}" -i "{temp_audio_path}" \
            -c:v copy -map 0:v:0 -map 1:a:0 -shortest -c:a aac \
            "{output_video_path}" -y'''
        
        subprocess.run(video_command, shell=True, check=True)
        
        # Check if output file exists and has a reasonable size
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 1000:
            print(f"✅ Success! Video created at: {output_video_path}")
            print(f"File size: {os.path.getsize(output_video_path) / (1024*1024):.2f} MB")
            
            # Display download link in Jupyter/Colab
            try:
                display(FileLink(output_video_path))
            except:
                pass
                
            return True
        else:
            print("❌ Something went wrong. Output file is missing or too small.")
            return False
            
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return False
    finally:
        # Optional: Clean up temporary files
        # if os.path.exists(temp_audio_path):
        #     os.remove(temp_audio_path)
        pass
