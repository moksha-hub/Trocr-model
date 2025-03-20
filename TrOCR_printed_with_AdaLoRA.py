import os
import warnings
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from evaluate import load
from peft import AdaLoraConfig, get_peft_model  # NEW: AdaLora imports

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Define dataset paths
DATASET_DIR = '/root/.cache/kagglehub/datasets/noorchauhan/rodrigo-spanish-text-17th-century/versions/1'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
TEXT_DIR = os.path.join(DATASET_DIR, 'text')
PARTITIONS_DIR = os.path.join(DATASET_DIR, 'partitions')
TRANSCRIPTIONS_FILE = os.path.join(TEXT_DIR, 'transcriptions.txt')

# Normalize transcriptions
def normalize_transcription(text):
    return text.replace('_', ' ').replace('ç', 'z')[:500]

# Load transcriptions from file
def load_transcriptions(trans_file):
    transcription_dict = {}
    with open(trans_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                transcription_dict[parts[0]] = normalize_transcription(parts[1])
            else:
                transcription_dict[parts[0]] = ""
    return transcription_dict

transcriptions = load_transcriptions(TRANSCRIPTIONS_FILE)

# Load partition IDs
def load_partition(file_path):
    with open(file_path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

train_ids = load_partition(os.path.join(PARTITIONS_DIR, 'train.txt'))
val_ids = load_partition(os.path.join(PARTITIONS_DIR, 'validation.txt'))
test_ids = load_partition(os.path.join(PARTITIONS_DIR, 'test.txt'))

# Build data samples and filter out empty transcriptions
all_image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.png')]
data_samples = [(img_id, os.path.join(IMAGES_DIR, img_file), transcriptions[img_id])
                for img_file in all_image_files
                if (img_id := os.path.splitext(img_file)[0]) in transcriptions and transcriptions[img_id].strip()]
print(f"Total paired samples (filtered): {len(data_samples)}")

# Filter samples based on partition lists
def filter_samples(samples, id_list):
    return [s for s in samples if any(s[0].startswith(pid) for pid in id_list)]

train_samples = filter_samples(data_samples, train_ids)
val_samples = filter_samples(data_samples, val_ids)
test_samples = filter_samples(data_samples, test_ids)

# Define image augmentation pipeline
def get_augmentation_pipeline(image_size=(256, 50)):
    return A.Compose([
        A.Resize(height=image_size[1], width=image_size[0], always_apply=True),
        A.PadIfNeeded(min_height=image_size[1], min_width=image_size[0], border_mode=0, value=0, always_apply=True),
        ToTensorV2()
    ])

# Define OCRDataset class with separate processing for images and text
class OCRDataset(Dataset):
    def __init__(self, samples, processor, image_size=(256, 50), max_text_length=512):
        self.samples = samples
        self.processor = processor
        self.transform = get_augmentation_pipeline(image_size)
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, img_path, transcription = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        image_tensor = self.transform(image=image)['image']

        # Use image_processor to process image
        pixel_values = self.processor.image_processor(
            images=image_tensor,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Process text using tokenizer with explicit max_length
        text_encoding = self.processor.tokenizer(
            transcription,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length
        )

        return {
            "pixel_values": pixel_values,
            "labels": text_encoding["input_ids"].squeeze(0)
        }

# Define custom data collator for dynamic padding
def data_collator(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    )
    return {
        "pixel_values": pixel_values,
        "labels": padded_labels
    }

# Load model and processor
model_name = "qantev/trocr-small-spanish"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Configure model
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

# NEW: Apply AdaLora configuration
peft_config = AdaLoraConfig(
    init_r=12,
    target_r=8,
    beta1=0.85,
    beta2=0.85,
    tinit=100,
    tfinal=1000,
    deltaT=10,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "query", "value"],  # Target attention layers in both encoder and decoder
    inference_mode=False,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Verify parameter efficiency

# Create datasets
train_dataset = OCRDataset(train_samples, processor, image_size=(256, 50))
val_dataset = OCRDataset(val_samples, processor, image_size=(256, 50))

torch.cuda.empty_cache()

# Load evaluation metrics
cer_metric = load("cer")
wer_metric = load("wer")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred_ids = np.argmax(logits, axis=-1)
    preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    refs = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    return {
        "cer": cer_metric.compute(predictions=preds, references=refs),
        "wer": wer_metric.compute(predictions=preds, references=refs)
    }

# Adjusted training args with slightly lower learning rate for AdaLora stability
training_args = TrainingArguments(
    output_dir="./trocr-small-finetuned",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    eval_strategy="epoch",
    eval_accumulation_steps=1,
    eval_do_concat_batches=False,
    save_strategy="epoch",
    logging_steps=30,
    learning_rate=5e-5,  # Slightly reduced from 3e-5 for AdaLora stability
    lr_scheduler_type="cosine",
    weight_decay=0.05,
    optim="adamw_torch",
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    save_total_limit=2,
    report_to=[],
    dataloader_pin_memory=True
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("num_items_in_batch", None)
        return super().compute_loss(model, inputs, return_outputs)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

torch.cuda.empty_cache()

# Save model and processor
model.save_pretrained("./trocr-small-finetuned")
processor.save_pretrained("./trocr-small-finetuned")
print("Fine-tuning complete and model saved.")
