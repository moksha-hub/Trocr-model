import os
import json
import random
import numpy as np
import pandas as pd
import torch
import xml.etree.ElementTree as ET

from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from datasets import Dataset

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator,
    EarlyStoppingCallback
)
from evaluate import load as load_metric

# -------------------------------
# 1. Helper function to parse XML and extract all bounding boxes plus metadata
# -------------------------------
def parse_xml(xml_path):
    """
    Parses the XML file to extract all <object> bounding boxes and image metadata.
    Returns a dict with:
      {
        "bboxes": [ {"name": ..., "bbox": (xmin, ymin, xmax, ymax)}, ... ],
        "filename": "100.jpg",
        "width": 5096,
        "height": 3296
      }
    If no bounding boxes found, returns None.
    """
    results = {"bboxes": []}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Optional: get filename and size info
        filename_node = root.find("filename")
        if filename_node is not None:
            results["filename"] = filename_node.text
        size_node = root.find("size")
        if size_node is not None:
            w = size_node.find("width")
            h = size_node.find("height")
            if w is not None and h is not None:
                results["width"] = int(w.text)
                results["height"] = int(h.text)

        # Parse all object bounding boxes
        for obj in root.findall("object"):
            obj_name = obj.find("name").text if obj.find("name") is not None else None
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                results["bboxes"].append({
                    "name": obj_name,
                    "bbox": (xmin, ymin, xmax, ymax)
                })
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
    if len(results["bboxes"]) == 0:
        return None
    return results

# -------------------------------
# 2. Load and process data by linking JSON records with XML info
# -------------------------------
def load_and_process_data():
    """
    Reads dataset/Labeled Data.json, finds matching images in dataset/Images,
    parses the corresponding XML files from dataset/Annotations, and unifies all
    bounding boxes into one (with added padding later during preprocessing).
    Returns a Hugging Face Dataset with columns:
      [image_path, text, id, xml_data]
    """
    annotation_file = os.path.join("dataset", "Labeled Data.json")
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", [])

    image_dir = os.path.join("dataset", "Images")
    xml_dir = os.path.join("dataset", "Annotations")

    valid_images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    # Map image number (basename without extension) to filename
    image_numbers = {os.path.splitext(f)[0]: f for f in valid_images}

    dataset_items = []
    for record in records:
        image_num = str(record.get("image_number", ""))
        image_filename = image_numbers.get(image_num, None)
        if image_filename and record.get("content"):
            image_path = os.path.join(image_dir, image_filename)
            text = record["content"]

            # Attempt to parse XML file
            xml_filename = image_num + ".xml"
            xml_path = os.path.join(xml_dir, xml_filename)
            xml_info = parse_xml(xml_path) if os.path.exists(xml_path) else None

            # If XML info exists, unify all bounding boxes into one enclosing box
            unified_bbox = None
            if xml_info:
                all_bboxes = xml_info["bboxes"]
                minx = min([b["bbox"][0] for b in all_bboxes])
                miny = min([b["bbox"][1] for b in all_bboxes])
                maxx = max([b["bbox"][2] for b in all_bboxes])
                maxy = max([b["bbox"][3] for b in all_bboxes])
                unified_bbox = (minx, miny, maxx, maxy)

            dataset_items.append({
                "image_path": image_path,
                "text": text,
                "id": record.get("id", ""),
                "xml_data": {
                    "filename": xml_info["filename"] if xml_info else None,
                    "width": xml_info["width"] if xml_info and "width" in xml_info else None,
                    "height": xml_info["height"] if xml_info and "height" in xml_info else None,
                    "unified_bbox": unified_bbox
                } if xml_info else None
            })

    df = pd.DataFrame(dataset_items)
    return Dataset.from_pandas(df)

full_dataset = load_and_process_data()
dataset_split = full_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]
print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

# -------------------------------
# 3. Enhanced image augmentations
# -------------------------------
augmentation_transforms = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)], p=0.6),
    transforms.RandomAffine(degrees=3, translate=(0.03, 0.03), shear=3, fill=255),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.2, 2.5))], p=0.4),
    transforms.RandomApply([transforms.RandomRotation(degrees=2)], p=0.3),
    transforms.Resize((384, 384))
])

# -------------------------------
# 4. Initialize processor and model
# -------------------------------
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.attention_dropout = 0.15
model.config.activation_dropout = 0.15

# Enable gradient checkpointing and disable cache for memory efficiency on T4 GPUs
model.gradient_checkpointing_enable()
model.config.use_cache = False

# -------------------------------
# 5. Preprocess function with bounding box cropping (with padding) and dynamic image/text processing
# -------------------------------
PADDING = 10  # Extra padding around bounding box

def preprocess_function(examples):
    images = []
    valid_indices = []
    for idx, path in enumerate(examples["image_path"]):
        try:
            img = Image.open(path).convert("RGB")
            xml_info = examples["xml_data"][idx]
            if xml_info and "unified_bbox" in xml_info and xml_info["unified_bbox"] is not None:
                xmin, ymin, xmax, ymax = xml_info["unified_bbox"]
                # Add padding (ensuring we remain within image bounds)
                xmin = max(0, xmin - PADDING)
                ymin = max(0, ymin - PADDING)
                xmax = min(img.width, xmax + PADDING)
                ymax = min(img.height, ymax + PADDING)
                img = img.crop((xmin, ymin, xmax, ymax))
            # Apply random invert
            if random.random() > 0.5:
                img = ImageOps.invert(img)
            # Apply enhanced transforms
            img = augmentation_transforms(img)
            # Optionally apply random sharpen
            if random.random() > 0.8:
                img = img.filter(ImageFilter.SHARPEN)
            images.append(img)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error loading {path}: {e} (Skipping)")
    valid_texts = [examples["text"][i] for i in valid_indices]
    if not valid_texts:
        return {}

    # Use fixed padding ("max_length") to ensure consistent tensor shapes
    encodings = processor(
        images=images,
        text=valid_texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    labels = encodings["labels"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    encodings["labels"] = labels
    return encodings

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

# -------------------------------
# 6. Updated training arguments (lightweight, T4 GPU compatible, better convergence)
# -------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="trocr_results_final",
    evaluation_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    fp16=True,
    learning_rate=2e-5,           # Lower learning rate for stability
    num_train_epochs=12,          # More epochs
    lr_scheduler_type="cosine",   # Cosine learning rate scheduler
    save_total_limit=3,
    save_steps=100,               # More frequent saves
    eval_steps=100,               # More frequent evaluations
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    report_to=["tensorboard"],
    weight_decay=0.01,
    push_to_hub=False,
    predict_with_generate=True,
    warmup_steps=800,             # More warmup steps
    max_grad_norm=0.8,            # Prevent exploding gradients
    adafactor=True,               # Adaptive optimizer for stability
    seed=42
)

# -------------------------------
# 7. Compute CER with text normalization
# -------------------------------
def normalize_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

def compute_metrics(pred):
    cer_metric = load_metric("cer")
    labels = pred.label_ids
    preds = pred.predictions

    preds = np.where(preds != -100, preds, processor.tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

    pred_str = processor.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)

    pred_str = [normalize_text(p) for p in pred_str]
    label_str = [normalize_text(l) for l in label_str]

    cer_val = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer_val}

# -------------------------------
# 8. Initialize Trainer with early stopping
# -------------------------------
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

# -------------------------------
# 9. Set seeds for reproducibility and train
# -------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

trainer.train()

# -------------------------------
# 10. Save the final model and processor
# -------------------------------
model.save_pretrained("trocr_final_model")
processor.save_pretrained("trocr_final_model")
print("Model saved to trocr_final_model/")

# Optional: Clean up CUDA cache
torch.cuda.empty_cache()
