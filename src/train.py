import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Import our dataset loading function
from dataset import prepare_dataset

@dataclass
class DataCollatorForSpeechSeq2SeqWithPadding:
    processor: Any
    reduction_factor: int = 1

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # Pad the text token input_ids
        batch = self.processor.tokenizer.pad(
            input_ids,
            padding=True,
            return_tensors="pt",
        )

        # Manually pad the mel spectrogram labels (shape per sample: [time_steps, 80])
        label_tensors = [torch.tensor(f["labels"], dtype=torch.float32) for f in features]

        # Truncate each label to be divisible by reduction_factor to avoid
        # off-by-one shape mismatches in the loss function
        if self.reduction_factor > 1:
            label_tensors = [t[: t.shape[0] - (t.shape[0] % self.reduction_factor)] for t in label_tensors]

        # pad_sequence pads along dim=0 (time axis); batch_first gives (batch, time, 80)
        padded_labels = torch.nn.utils.rnn.pad_sequence(label_tensors, batch_first=True, padding_value=-100.0)

        # Build a decoder attention mask: 1 for real frames, 0 for padded frames
        max_len = padded_labels.shape[1]
        decoder_attention_mask = torch.zeros(len(features), max_len, dtype=torch.long)
        for i, t in enumerate(label_tensors):
            decoder_attention_mask[i, : t.shape[0]] = 1

        batch["labels"] = padded_labels
        batch["decoder_attention_mask"] = decoder_attention_mask
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch

def train():
    print("Loading prepared dataset...")
    dataset = prepare_dataset()
    if dataset is None or len(dataset) == 0:
        print("Dataset is empty. Please record audio first.")
        return

    # Split dataset safely
    if len(dataset) < 5:
        print("WARNING: Dataset is extremely small. Finetuning may severely overfit.")
        train_ds = dataset
        eval_ds = dataset
    else:
        split_ds = dataset.train_test_split(test_size=0.1)
        train_ds = split_ds["train"]
        eval_ds = split_ds["test"]

    print("Loading SpeechT5 Model and Processor...")
    checkpoint = "microsoft/speecht5_tts"
    processor = SpeechT5Processor.from_pretrained(checkpoint)
    model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)

    # Disable cache during training
    model.config.use_cache = False

    data_collator = DataCollatorForSpeechSeq2SeqWithPadding(
        processor=processor,
        reduction_factor=model.config.reduction_factor,
    )

    # Setup training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="models/speecht5_finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=500, # Adjust longer based on quality later!
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps", # HF Transformers 4.41 deprecates evaluation_strategy for eval_strategy
        per_device_eval_batch_size=2,
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=processor,
    )

    print("Starting Training...")
    trainer.train()

    # Save final model state
    model.save_pretrained("models/speecht5_finetuned_final")
    processor.save_pretrained("models/speecht5_finetuned_final")
    print("Training complete and models saved to models/speecht5_finetuned_final!")

if __name__ == "__main__":
    train()
