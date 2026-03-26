import gradio as gr
import os
import csv
import uuid
import shutil


# Directory to save the dataset
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "my_voice_dataset")
AUDIO_DIR = os.path.join(DATA_DIR, "wavs")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)

# Initialize metadata.csv with header if it doesn't exist
if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, mode='w', newline='', encoding='utf-8') as f:
        # Standard LJSpeech format uses '|' delimiter
        writer = csv.writer(f, delimiter='|')
        writer.writerow(["file_name", "transcription"])

def count_recordings():
    if not os.path.exists(METADATA_FILE):
        return 0
    with open(METADATA_FILE, mode='r', encoding='utf-8') as f:
        # subtract 1 for the header
        return max(0, sum(1 for line in f) - 1)

def save_recording(audio_path, transcription):
    if not audio_path or not transcription.strip():
        return "Please provide both an audio recording and a transcription."
    
    # Generate a unique filename for the new recording
    unique_id = str(uuid.uuid4())[:8]
    file_name = f"sample_{unique_id}.wav"
    dest_path = os.path.join(AUDIO_DIR, file_name)
    
    # Copy the audio file from Gradio's temp directory to our dataset directory
    shutil.copy2(audio_path, dest_path)
    
    # Save the transcription to metadata.csv
    with open(METADATA_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow([file_name, transcription.strip()])
        
    return f"Saved successfully! Total recordings: {count_recordings()}"

with gr.Blocks(title="Voice Dataset Recorder") as demo:
    gr.Markdown("# 🎙️ Record Your Voice for TTS Fine-Tuning")
    gr.Markdown("Type the text you want to read aloud, click record, and save it directly to your dataset.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text to Read (Transcription)", placeholder="Type the text here...", lines=3)
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record your voice")
            save_btn = gr.Button("Save to Dataset", variant="primary")
        
        with gr.Column():
            output_msg = gr.Textbox(label="Status", value=f"Total recordings in dataset: {count_recordings()}", interactive=False)
            
    # When user clicks save, process the inputs and show the output status message
    save_btn.click(
        fn=save_recording,
        inputs=[audio_input, text_input],
        outputs=[output_msg]
    )

    # Optional: Clear the text box and audio component after saving to prepare for the next recording
    def clear_ui():
        return None, ""
    
    save_btn.click(fn=clear_ui, inputs=[], outputs=[audio_input, text_input])

if __name__ == "__main__":
    print(f"Saving dataset to: {DATA_DIR}")
    demo.launch()
