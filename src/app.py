import os
import glob
import sys
import torch
import librosa

# Monkey patch torchaudio just in case for speechbrain compatibility
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

# Monkey patch huggingface_hub for offline inference with speechbrain
import huggingface_hub
from huggingface_hub.utils import EntryNotFoundError
_original_download = huggingface_hub.hf_hub_download
def _patched_download(*args, **kwargs):
    kwargs.pop('use_auth_token', None)
    try:
        return _original_download(*args, **kwargs)
    except EntryNotFoundError as e:
        if 'custom.py' in str(e) or 'custom.py' in kwargs.get('filename', ''):
            dummy_path = os.path.join(kwargs.get('cache_dir') or 'models', 'custom.py')
            os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
            with open(dummy_path, 'w') as f:
                f.write('')
            return dummy_path
        raise e
huggingface_hub.hf_hub_download = _patched_download

import gradio as gr
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.inference.speaker import EncoderClassifier

print("Loading Models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model_dir = "models/speecht5_finetuned_final"

if not os.path.exists(model_dir):
    print(f"Error: Could not find trained model at {model_dir}. Did you unzip the downloaded trained_model.zip correctly?")
    sys.exit(1)

# Load the fine-tuned model and processor
processor = SpeechT5Processor.from_pretrained(model_dir)
model = SpeechT5ForTextToSpeech.from_pretrained(model_dir).to(device)

# Load the vocoder (converts spectrogram to audio)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

# Load the speaker embedding extraction model
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name, 
    run_opts={"device": device}, 
    savedir=os.path.join("models", spk_model_name)
)

print("Models loaded successfully!")

def generate_speech(text, reference_audio_path=None):
    if not text.strip():
        return None, "Please enter some text."
        
    # Process text
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # If no reference audio is provided, try to find one in the dataset
    if reference_audio_path is None:
        dataset_wavs = glob.glob("data/my_voice_dataset/wavs/*.wav")
        if not dataset_wavs:
            return None, "No reference audio found. Please record or upload a sample."
        reference_audio_path = dataset_wavs[0]

    # Extract speaker embedding
    audio, sr = librosa.load(reference_audio_path, sr=16000)
    waveform = torch.tensor(audio).unsqueeze(0).to(device)
    
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(waveform)
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().unsqueeze(0).to(device) # shape: (1, 512)

    # Generate speech
    with torch.no_grad():
        speech = model.generate(
            input_ids,
            speaker_embeddings=speaker_embeddings,
            vocoder=vocoder,
        )

    # The output is a 1D tensor of audio samples at 16kHz
    audio_output = speech.cpu().numpy()
    
    return (16000, audio_output), f"Generated using voice reference: {os.path.basename(reference_audio_path)}"

# Create Gradio Interface
with gr.Blocks(title="My Voice Clone") as demo:
    gr.Markdown("# 🗣️ My Fine-Tuned Voice Clone")
    gr.Markdown("Type text below and generate speech using your custom model. By default, it will use an audio sample from your dataset to emulate your voice characteristics, or you can upload/record a specific reference audio to match a specific tone/emotion!")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text to generate", lines=3, placeholder="Hello world! This is my digitally cloned voice doing the talking now.")
            ref_audio = gr.Audio(label="Reference Audio (Optional - uses dataset by default)", type="filepath")
            generate_btn = gr.Button("Generate Cloned Speech", variant="primary")
            
        with gr.Column():
            audio_output = gr.Audio(label="Generated Audio")
            status_text = gr.Textbox(label="Status", interactive=False)
            
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, ref_audio],
        outputs=[audio_output, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)
