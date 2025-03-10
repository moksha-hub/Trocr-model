from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
from evaluate import load as load_metric
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageOps
import json
import random
import torch
from torchvision.transforms import ColorJitter

# Initialize processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Configure model settings to reduce overfitting
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.attention_dropout = 0.1  # Add dropout for regularization
model.config.activation_dropout = 0.1

# Load and process dataset
def load_and_process_data():
    annotation_file = os.path.join("dataset", "Labeled Data.json")
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get('records', [])
    
    # Extract valid .jpg images
    image_dir = os.path.join("dataset", "Images")
    valid_images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg"))]
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

# Load and split dataset
full_dataset = load_and_process_data()
dataset_split = full_dataset.train_test_split(
    test_size=0.1,
    shuffle=True,
    seed=42
)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Preprocessing with data augmentation
def preprocess_function(examples):
    images = []
    valid_indices = []
    for idx, path in enumerate(examples["image_path"]):
        try:
            img = Image.open(path).convert("RGB")
            
            # Data augmentation
            if random.random() > 0.5:
                img = ImageOps.invert(img)  # Invert colors (common for old manuscripts)
            if random.random() > 0.5:
                img = ColorJitter(brightness=0.2, contrast=0.2)(img)  # Adjust brightness/contrast
            angle = random.uniform(-2, 2)
            img = img.rotate(angle, expand=False)
            img = img.resize((384, 384), Image.BILINEAR)  # Ensure model input size
            
            images.append(img)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error loading {path}: {str(e)} (Skipping)")
    
    # Filter valid texts
    valid_texts = [examples["text"][i] for i in valid_indices]
    if not valid_texts:
        return {}
    
    # Process with the processor
    encodings = processor(
        images,
        text=valid_texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Replace padding tokens with -100
    labels = encodings["labels"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    encodings["labels"] = labels
    
    return encodings

# Apply preprocessing
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=4,
    remove_columns=train_dataset.column_names,
    num_proc=1
)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=4,
    remove_columns=eval_dataset.column_names,
    num_proc=1
)

# Training arguments (fixed save_steps)
training_args = Seq2SeqTrainingArguments(
    output_dir="spanish_trocr_results",
    evaluation_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    fp16=True,
    learning_rate=5e-5,
    num_train_epochs=5,
    save_total_limit=3,
    save_steps=200,  # Must be a multiple of eval_steps=200
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
    adafactor=True,
    seed=42
)

# Compute metrics function
def compute_metrics(pred):
    metric = load_metric("cer")
    labels = pred.label_ids
    preds = pred.predictions
    
    # Replace -100 with pad_token_id for decoding
    preds = np.where(preds != -100, preds, processor.tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    
    pred_str = processor.batch_decode(
        preds,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
        num_beams=5  # Beam search for better predictions
    )
    
    label_str = processor.batch_decode(
        labels,
        skip_special_tokens=True
    )
    
    return {"cer": metric.compute(predictions=pred_str, references=label_str)}

# Initialize trainer with EarlyStoppingCallback
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

# Train!
trainer.train()

# Save best model
model.save_pretrained("spanish_trocr_final")
processor.save_pretrained("spanish_trocr_final")
print("Model saved to spanish_trocr_final/")

# Optional: Clean up CUDA cache
torch.cuda.empty_cache()
