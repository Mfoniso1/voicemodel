import os
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

def download_and_save_models():
    # Base directory to save models
    save_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(save_directory, exist_ok=True)
    
    # 1. Download and save the Processor
    print("Downloading Processor...")
    processor_dir = os.path.join(save_directory, "speecht5_tts_processor")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    processor.save_pretrained(processor_dir)
    print(f"Processor saved to {processor_dir}")
    
    # 2. Download and save the TTS Model
    print("\nDownloading TTS Model...")
    model_dir = os.path.join(save_directory, "speecht5_tts_model")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    model.save_pretrained(model_dir)
    print(f"Model saved to {model_dir}")
    
    # 3. Download and save the Vocoder (HiFi-GAN)
    print("\nDownloading Vocoder (HifiGan)...")
    vocoder_dir = os.path.join(save_directory, "speecht5_hifigan_vocoder")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    vocoder.save_pretrained(vocoder_dir)
    print(f"Vocoder saved to {vocoder_dir}")
    
    print("\nDownload complete! Models are ready for fine-tuning.")

if __name__ == "__main__":
    download_and_save_models()
