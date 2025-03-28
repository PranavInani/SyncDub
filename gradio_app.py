import gradio as gr
import os
from pathlib import Path

def create_gradio_interface():
    # Define available languages and their full names
    LANGUAGES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese"
    }

    # List of supported file extensions
    SUPPORTED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

    def process_video(
        media_input,
        target_language,
        tts_engine,
        max_speakers,
        *speaker_inputs
    ):
        # Create voice config from inputs
        voice_config = {}
        
        if tts_engine == "Edge TTS":
            for i in range(max_speakers):
                gender = speaker_inputs[i]
                voice_config[i] = {
                    'engine': 'edge_tts',
                    'gender': gender.lower()
                }
        else:
            for i in range(max_speakers):
                ref_audio = speaker_inputs[i]
                voice_config[i] = {
                    'engine': 'xtts',
                    'reference_audio': ref_audio.name if ref_audio else None
                }

        # Here you would call your existing pipeline functions
        # For demonstration, we'll just return a dummy video
        # Replace this with your actual processing logic
        output_path = "path/to/processed_video.mp4"
        
        return output_path

    def update_speaker_inputs(max_speakers, tts_engine):
        speaker_components = []
        for i in range(int(max_speakers)):
            if tts_engine == "Edge TTS":
                speaker_components.append(
                    gr.Dropdown(label=f"Speaker {i+1} Gender", 
                              choices=["Male", "Female"], visible=True)
                )
            else:
                speaker_components.append(
                    gr.File(label=f"Speaker {i+1} Reference Audio", 
                          file_types=["audio"], visible=True)
                )
        return speaker_components

    with gr.Blocks(title="Video Dubbing Pipeline", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Video Dubbing Pipeline")
        gr.Markdown("Upload a video or provide a URL, then select translation and voice options.")

        with gr.Row():
            with gr.Column():
                media_input = gr.Textbox(
                    label="Video URL or File Path",
                    placeholder="Enter URL or upload file below",
                )
                media_upload = gr.File(
                    label="Upload Video File",
                    file_types=SUPPORTED_EXTENSIONS,
                )
                target_language = gr.Dropdown(
                    label="Target Language",
                    choices=list(LANGUAGES.values()),
                    value="Spanish"
                )
                tts_engine = gr.Radio(
                    label="TTS Engine",
                    choices=["Edge TTS", "XTTS"],
                    value="Edge TTS"
                )
                max_speakers = gr.Slider(
                    label="Maximum Number of Speakers",
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=1
                )

                speaker_inputs = gr.Column()

                submit_btn = gr.Button("Generate Dubbed Video", variant="primary")

            with gr.Column():
                output_video = gr.Video(label="Dubbed Video")
                status = gr.Textbox(label="Processing Status", interactive=False)

        # Dynamic speaker inputs
        max_speakers.change(
            fn=update_speaker_inputs,
            inputs=[max_speakers, tts_engine],
            outputs=speaker_inputs
        )

        tts_engine.change(
            fn=lambda engine, max_spk: update_speaker_inputs(max_spk, engine),
            inputs=[tts_engine, max_speakers],
            outputs=speaker_inputs
        )

        submit_btn.click(
            fn=process_video,
            inputs=[
                media_input,
                target_language,
                tts_engine,
                max_speakers,
                speaker_inputs
            ],
            outputs=output_video,
            api_name="dub_video"
        )

    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)