import os
import torch
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

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

from datasets import load_dataset, Audio
from transformers import SpeechT5Processor

def prepare_dataset():
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "my_voice_dataset")
    METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
    AUDIO_DIR = os.path.join(DATA_DIR, "wavs")
    
    if not os.path.exists(METADATA_FILE):
        print(f"No metadata found at {METADATA_FILE}. Have you recorded data yet?")
        return None

    print(f"Loading metadata from {METADATA_FILE}...")
    dataset = load_dataset("csv", data_files=METADATA_FILE, delimiter="|", split="train")

    print("Mapping audio files and enforcing 16kHz sampling rate...")
    import librosa
    def map_audio_path(batch):
        audio_path = os.path.join(AUDIO_DIR, batch["file_name"])
        # Use librosa to decode and resample, bypassing datasets torchcodec backend
        array, sr = librosa.load(audio_path, sr=16000)
        batch["audio"] = {"array": array, "sampling_rate": sr}
        return batch

    dataset = dataset.map(map_audio_path)
    
    print("Loading SpeechT5 tokenizer and feature extractor...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    
    print("Loading SpeechBrain x-vector model for speaker embeddings...")
    from speechbrain.inference.speaker import EncoderClassifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
    speaker_model = EncoderClassifier.from_hparams(
        source=spk_model_name, 
        run_opts={"device": device}, 
        savedir=os.path.join("models", spk_model_name)
    )

    def extract_all_features(batch):
        # 1. Audio Processing
        audio = batch["audio"]
        
        # 2. Text tokenization + Mel-spectrogram generation
        processed = processor(
            text=batch["transcription"],
            audio_target=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=False,
        )
        
        # 3. Create Speaker Embedding
        # The x-vector model expects waveform tensor
        waveform = torch.tensor(audio["array"]).unsqueeze(0)
        with torch.no_grad():
            speaker_embeddings = speaker_model.encode_batch(waveform)
            # L2 normalization over the embedding dimension
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()

        # Update batch with processed targets
        batch["labels"] = processed["labels"][0]
        batch["input_ids"] = processed["input_ids"]
        batch["speaker_embeddings"] = speaker_embeddings
        
        return batch

    print("Processing audio sequences and generating text tokens & speaker embeddings...")
    # map runs sequence by sequence
    dataset = dataset.map(
        extract_all_features,
        remove_columns=dataset.column_names,
    )
    
    print("Dataset preparation complete!")
    return dataset

if __name__ == "__main__":
    ds = prepare_dataset()
    if ds is not None:
        print("Final Dataset object:")
        print(ds)
