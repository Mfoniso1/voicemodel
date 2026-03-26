# To-Do: Fine-Tuning SpeechT5

## Phase 1: Environment Setup
- [ ] **Install dependencies**: Open your terminal in the `voicemodel` folder and run `pip install -r requirements.txt`.
- [ ] **Download Base Models**: Run `python src/download_models.py` to cache the 1.5GB SpeechT5 model and Vocoder locally.

## Phase 2: Data Collection
- [ ] **Record Voice Samples**: Run `python src/record_data.py` in your terminal.
- [ ] **Build Dataset**: Open the web link (`http://127.0.0.1:7860`) and use your microphone to record at least 50-100 high-quality sentences. 

## Phase 3: Data Preprocessing
- [ ] **Load Dataset**: Implement `src/dataset.py` to load the `metadata.csv` and audio files using the Hugging Face `datasets` library.
- [ ] **Process Audio**: Resample all your `.wav` files to 16kHz and convert the text into tokens using `SpeechT5Processor`.
- [ ] **Generate Speaker Embeddings**: SpeechT5 requires a numerical representation of your vocal characteristics (an "x-vector"). We will need to use a model like `speechbrain/spkrec-xvect-voxceleb` to extract this embedding from one of your audio clips.

## Phase 4: Model Training (Fine-Tuning)
- [ ] **Setup Trainer**: Implement `src/train.py` using `transformers.Trainer`.
- [ ] **Configure Training Arguments**: Set batch sizes, learning rates, and save directories.
- [ ] **Run Training**: Execute `python src/train.py` in your terminal and wait for the model to update its weights based on your voice.

## Phase 5: Testing & Inference
- [ ] **Update Inference App**: Modify `src/app.py` to point to your new fine-tuned checkpoint in the `models/` directory.
- [ ] **Launch Web Interface**: Run `python src/app.py`.
- [ ] **Test Your Avatar Voice**: Type in new sentences, generate the text-to-speech outloud, and listen to your newly cloned digital voice!
