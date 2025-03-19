from pyannote.audio import Pipeline
import torch
import os
import time

class SpeakerDiarizer:
    def __init__(self, hf_token, device=None):
        """Initialize speaker diarization with HuggingFace token"""
        self.diarization_pipeline = None
        try:
            print("Loading diarization pipeline...")
            # Check available devices
            if device is None:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Use the newer version that's compatible with your libraries
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            self.diarization_pipeline.to(torch.device(device))
            print("Diarization model loaded successfully!")
        except Exception as e:
            print(f"Error loading diarization model: {e}")
    
    def diarize(self, audio_path, min_speakers=1, max_speakers=None, device=None):
        """Identify speakers in audio file"""
        if not self.diarization_pipeline:
            print("Diarization pipeline not available")
            return []
        
        try:
            print("Starting speaker diarization (this may take several minutes for longer files)...")
            start_time = time.time()
            
            # Set parameters for diarization
            params = {}
            if min_speakers is not None:
                params["min_speakers"] = min_speakers
            if max_speakers is not None:
                params["max_speakers"] = max_speakers
            
            # Set device if specified (cuda:0, cpu, etc.)
            if device:
                print(f"Using device: {device}")
                self.diarization_pipeline.to(torch.device(device))
            
            # Add progress updates
            print("Running diarization model...")
            print("This process may take several minutes with no visible progress...")
            print("Consider using a smaller audio segment for testing")
            
            # Use the diarization pipeline
            diarization = self.diarization_pipeline(audio_path, **params)
            
            print("Processing diarization results...")
            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
            
            # Encode speaker IDs consistently (SPEAKER_00, SPEAKER_01, etc.)
            speakers = self._reencode_speakers(speakers)
            print(f"Diarization completed in {time.time() - start_time:.1f}s")
            print(f"Found {len(set(s['speaker'] for s in speakers))} speakers")
            
            return speakers
        except Exception as e:
            print(f"Error during diarization: {e}")
            return []
    
    def assign_speakers_to_segments(self, segments, speakers):
        """Match transcription segments with speaker information"""
        for segment in segments:
            segment_mid = (segment["start"] + segment["end"]) / 2
            # Find speaker active at this time
            for speaker_turn in speakers:
                if speaker_turn["start"] <= segment_mid <= speaker_turn["end"]:
                    segment["speaker"] = speaker_turn["speaker"]
                    break
            else:
                segment["speaker"] = "unknown"
                
        return segments

    def _reencode_speakers(self, speakers):
        """Ensure consistent speaker IDs (SPEAKER_00, SPEAKER_01, etc.)"""
        if not speakers:
            return speakers
            
        # Get unique speaker IDs
        unique_speakers = set()
        for turn in speakers:
            unique_speakers.add(turn['speaker'])
        
        # Create mapping
        speaker_map = {old_id: f"SPEAKER_{i:02d}" 
                    for i, old_id in enumerate(sorted(unique_speakers))}
        
        # Apply mapping
        for turn in speakers:
            turn['speaker'] = speaker_map[turn['speaker']]
        
        return speakers