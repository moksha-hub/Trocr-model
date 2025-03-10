from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator
)
from datasets import Dataset, load_dataset
from evaluate import load as load_metric
import numpy as np
import pandas as pd
import os
from PIL import Image
import json

# Initialize processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Configure model
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

# Load and process dataset
def load_and_process_data():
    # Load JSON data
    annotation_file = os.path.join("dataset", "Labeled Data.json")
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get('records', [])
    
    # Extract valid image filenames (only .jpg)
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
    
    # Create Dataset from filtered items
    df = pd.DataFrame(dataset_items)
    dataset = Dataset.from_pandas(df)
    return dataset

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

# Preprocessing function with error handling
def preprocess_function(examples):
    images = []
    valid_indices = []
    for idx, path in enumerate(examples["image_path"]):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error loading {path}: {str(e)} (Skipping)")

    # Filter texts corresponding to valid images
    valid_texts = [examples["text"][i] for i in valid_indices]
    
    if len(valid_texts) == 0:
        return {}  # Return empty if no valid examples
    
    encodings = processor(
        images,
        text=valid_texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Replace padding token with -100
    labels = encodings["labels"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    encodings["labels"] = labels
    
    return encodings

# Apply preprocessing (no parallel processing)
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=2,
    remove_columns=train_dataset.column_names,
    num_proc=1
)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=2,
    remove_columns=eval_dataset.column_names,
    num_proc=1
)

# Training arguments (added predict_with_generate=True)
training_args = Seq2SeqTrainingArguments(
    output_dir="spanish_trocr_results",
    evaluation_strategy="steps",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    fp16=True,
    learning_rate=3e-5,
    num_train_epochs=2,
    save_total_limit=2,
    save_steps=50,
    eval_steps=50,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    report_to=["tensorboard"],
    weight_decay=0.01,
    push_to_hub=False,
    predict_with_generate=True  # <--- Added this line
)

# ... [previous code remains the same]

def compute_metrics(pred):
    metric = load_metric("cer")
    labels = pred.label_ids
    preds = pred.predictions
    
    # Replace -100 with pad_token_id for decoding
    preds = np.where(preds != -100, preds, processor.tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    
    pred_str = processor.batch_decode(preds, skip_special_tokens=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)
    
    # ✅ Fix: Remove ["cer"], since `metric.compute()` returns the float directly
    return {"cer": metric.compute(predictions=pred_str, references=label_str)}

# ... [rest of the code remains the same]

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics
)

# Train!
trainer.train()

# Save final model
model.save_pretrained("spanish_trocr_final")
processor.save_pretrained("spanish_trocr_final")
print("Model saved to spanish_trocr_final/")
