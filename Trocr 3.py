from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator,
    EarlyStoppingCallback
)
from datasets import Dataset
from evaluate import load as load_metric
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageOps, ImageFilter
import json
import random
import torch
from torchvision import transforms

# Use the base model to save on computation:
MODEL_CHECKPOINT = "microsoft/trocr-base-handwritten"

# Initialize processor and model
processor = TrOCRProcessor.from_pretrained(MODEL_CHECKPOINT)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_CHECKPOINT)

# Update model config for better regularization
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.attention_dropout = 0.15
model.config.activation_dropout = 0.15

# Load and process dataset (adapted for SpanishNotaryCollection)
def load_and_process_data():
    annotation_file = os.path.join("dataset", "Labeled Data.json")
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get('records', [])

    image_dir = os.path.join("dataset", "Images")
    valid_images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_numbers = {os.path.splitext(f)[0]: f for f in valid_images}

    dataset_items = []
    for record in records:
        image_number = str(record.get('image_number', ''))
        image_filename = image_numbers.get(image_number, None)
        if image_filename and record.get('content'):
            image_path = os.path.join(image_dir, image_filename)
            text = record['content']
            dataset_items.append({
                'image_path': image_path,
                'text': text,
                'id': record['id']
            })

    df = pd.DataFrame(dataset_items)
    return Dataset.from_pandas(df)

full_dataset = load_and_process_data()
dataset_split = full_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Define a streamlined set of image augmentations tailored for historical notary documents.
augmentation_transforms = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
    transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), shear=2, fill=255),
    transforms.RandomApply([transforms.Lambda(lambda img: ImageOps.autocontrast(img))], p=0.3),
    transforms.Resize((384, 384))  # Consider reducing to (256, 256) if needed.
])

def preprocess_function(examples):
    images = []
    valid_indices = []
    for idx, path in enumerate(examples["image_path"]):
        try:
            img = Image.open(path).convert("RGB")
            img = augmentation_transforms(img)
            # Optionally, add a low-probability sharpen filter
            if random.random() > 0.9:
                img = img.filter(ImageFilter.SHARPEN)
            images.append(img)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error loading {path}: {e} (Skipping)")

    valid_texts = [examples["text"][i] for i in valid_indices]
    if not valid_texts:
        return {}

    encodings = processor(
        images,
        text=valid_texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    labels = encodings["labels"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    encodings["labels"] = labels
    return encodings

# Map the preprocessing function to the datasets
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=2,  # Reduced batch size for memory efficiency
    remove_columns=train_dataset.column_names,
    num_proc=1
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=2,  # Reduced batch size for memory efficiency
    remove_columns=eval_dataset.column_names,
    num_proc=1
)

# Adjust training arguments to use smaller batches
training_args = Seq2SeqTrainingArguments(
    output_dir="trocr_results_improved",
    evaluation_strategy="steps",
    per_device_train_batch_size=2,  # Lowered batch size
    per_device_eval_batch_size=2,   # Lowered batch size
    gradient_accumulation_steps=2,
    fp16=True,
    learning_rate=3e-5,
    num_train_epochs=10,
    save_total_limit=3,
    save_steps=200,
    eval_steps=200,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    report_to=["tensorboard"],
    weight_decay=0.01,
    push_to_hub=False,
    predict_with_generate=True,
    warmup_steps=500,
    max_grad_norm=1.0,
    adafactor=False,
    seed=42
)

# Compute CER using beam search decoding (num_beams=10)
def compute_metrics(pred):
    metric = load_metric("cer")
    labels = pred.label_ids
    preds = pred.predictions
    preds = np.where(preds != -100, preds, processor.tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

    pred_str = processor.batch_decode(
        preds,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
        num_beams=10
    )
    label_str = processor.batch_decode(
        labels,
        skip_special_tokens=True
    )
    return {"cer": metric.compute(predictions=pred_str, references=label_str)}

# Initialize the trainer with early stopping
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )]
)

# Start training
trainer.train()

# Save the best model and processor
model.save_pretrained("trocr_final_improved")
processor.save_pretrained("trocr_final_improved")
print("Model saved to 'trocr_final_improved/'.")

# Optional: Clean up CUDA cache
torch.cuda.empty_cache()
