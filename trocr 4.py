import warnings
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
import xml.etree.ElementTree as ET
import json
import random
import torch
from torchvision import transforms

# Suppress specific warnings
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

MODEL_CHECKPOINT = "microsoft/trocr-base-handwritten"

# Initialize processor and model in online mode
processor = TrOCRProcessor.from_pretrained(MODEL_CHECKPOINT)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_CHECKPOINT)

# Configure beam search and model regularization
model.config.decoder.num_beams = 10
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.attention_dropout = 0.15
model.config.activation_dropout = 0.15
model.config.use_cache = False

##############################
# XML Preprocessing Logic
##############################
def parse_xml_annotation(xml_path):
    """
    Parses a Pascal VOC style XML file and extracts the filename and a list of annotations.
    Each annotation is a dict with 'text' (label) and 'bbox' (xmin, ymin, xmax, ymax).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
    annotations = []
    for obj in root.findall("object"):
        # Split by comma and take the last part, then strip whitespace
        name = obj.find("name").text.split(",")[-1].strip()
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        annotations.append({
            "text": name,
            "bbox": (xmin, ymin, xmax, ymax)
        })
    return filename, annotations

def load_and_process_data():
    """
    Loads XML annotation files from the dataset/Images directory.
    For each XML, it finds the corresponding .jpg image and processes each annotation
    as a separate sample with its image path, text label, and bounding box.
    """
    dataset_dir = "dataset"
    image_dir = os.path.join(dataset_dir, "Images")
    
    dataset_items = []
    for file in os.listdir(image_dir):
        if file.endswith(".xml"):
            xml_path = os.path.join(image_dir, file)
            img_file = file.replace('.xml', '.jpg')
            image_path = os.path.join(image_dir, img_file)
            if not os.path.exists(image_path):
                continue
            _, annotations = parse_xml_annotation(xml_path)
            for anno in annotations:
                dataset_items.append({
                    "image_path": image_path,
                    "text": anno["text"],
                    "bbox": anno["bbox"]
                })
    df = pd.DataFrame(dataset_items)
    return Dataset.from_pandas(df)

full_dataset = load_and_process_data()
dataset_split = full_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

##############################
# Data Augmentations and Preprocessing
##############################
augmentation_transforms = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
    transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), shear=2, fill=255),
    transforms.RandomApply([transforms.Lambda(lambda img: ImageOps.autocontrast(img))], p=0.3),
    transforms.Resize((384, 384))  # Consider reducing to (256,256) if needed
])

def preprocess_function(examples):
    images = []
    valid_indices = []
    for idx, path in enumerate(examples["image_path"]):
        try:
            img = Image.open(path).convert("RGB")
            # Crop using bounding box from the XML annotation
            img = img.crop(examples["bbox"][idx])
            img = augmentation_transforms(img)
            # Optionally, add a low-probability sharpen filter
            if random.random() > 0.9:
                img = img.filter(ImageFilter.SHARPEN)
            images.append(img)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error processing {examples['image_path'][idx]}: {e}")
    valid_texts = [examples["text"][i] for i in valid_indices]
    if not valid_texts:
        return {}
    encodings = processor(
        images=images,
        text=valid_texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    encodings["labels"] = torch.where(encodings["labels"] == processor.tokenizer.pad_token_id,
                                      -100, encodings["labels"])
    return encodings

# Map the preprocessing function without multiprocessing
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=2,
    remove_columns=train_dataset.column_names
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=2,
    remove_columns=eval_dataset.column_names
)

##############################
# Training Arguments and Trainer Setup
##############################
training_args = Seq2SeqTrainingArguments(
    output_dir="trocr_results_improved",
    evaluation_strategy="steps",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
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
    warmup_steps=500,
    weight_decay=0.01,
    optim="adafactor",  # Using AdaFactor for efficiency
    seed=42,
    dataloader_num_workers=2,
    report_to=["tensorboard"],
    predict_with_generate=True
)

def compute_metrics(pred):
    metric = load_metric("cer")
    labels = pred.label_ids
    preds = pred.predictions
    preds = np.where(preds != -100, preds, processor.tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    pred_str = processor.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True, num_beams=10)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)
    return {"cer": metric.compute(predictions=pred_str, references=label_str)}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
)

##############################
# Training and Saving the Model
##############################
trainer.train()
model.save_pretrained("trocr_final_improved")
processor.save_pretrained("trocr_final_improved")
print("Model saved to 'trocr_final_improved/'.")
torch.cuda.empty_cache()

