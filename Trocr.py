import os
import json
import numpy as np
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset, Features, Value, Array3D
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# 1. Set dataset paths
image_folder = "SpanishNotaryCollection/dataset/Images"
json_path = "SpanishNotaryCollection/dataset/Labeled Data.json"

# 2. Load JSON file
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. Extract records
records = data.get("records", [])

# 4. Create mapping {image_path: text}
image_text_map = {}
for record in records:
    image_number = record.get("image_number")
    text = record.get("content")
    if image_number is not None and text:
        img_file = f"{image_number}.jpg"
        img_path = os.path.join(image_folder, img_file)
        if os.path.exists(img_path):
            image_text_map[img_path] = text

# 5. Function to load and process images
def process_image(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        image = image.resize((384, 384))
        img_array = np.array(image, dtype=np.uint8)
        if img_array.shape != (384, 384, 3):
            print(f"❌ Invalid shape for {img_path}: {img_array.shape}")
            return None
        return img_array  # Return NumPy array
    except Exception as e:
        print(f"❌ Error loading {img_path}: {e}")
        return None

# 6. Load images in parallel
image_paths = list(image_text_map.keys())
with ThreadPoolExecutor(max_workers=8) as executor:
    images = list(executor.map(process_image, image_paths))

# 7. Remove failed loads
valid_data = [(img, image_text_map[path]) for img, path in zip(images, image_paths) if img is not None]
if not valid_data:
    raise ValueError("❌ No valid images found.")

# 8. Ensure all images are NumPy arrays
valid_images = []
for img, _ in valid_data:
    if isinstance(img, list):  # Convert lists to NumPy arrays if needed
        img = np.array(img, dtype=np.uint8)
    valid_images.append(img)
texts = [text for _, text in valid_data]

# 9. Create dataset with explicit feature definitions
dataset = Dataset.from_dict(
    {"image": valid_images, "text": texts},
    features=Features({"image": Array3D(dtype="uint8", shape=(384, 384, 3)), "text": Value("string")}),
)
print(f"✅ Dataset successfully created with {len(dataset)} samples!")

# 10. Load TrOCR model and processor with use_fast=True
model_name = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name, use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# 11. Configure model for text generation
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# 12. Split dataset into train and validation
train_val_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_split["train"]
eval_dataset = train_val_split["test"]

# 13. Preprocess function (batched=True compatible)
def preprocess_data(examples):
    # Ensure each image is a NumPy array before conversion
    images = []
    for img in examples["image"]:
        if isinstance(img, list):
            img = np.array(img, dtype=np.uint8)
        images.append(Image.fromarray(img))  # Convert to PIL Image
    
    encoding = processor(images, return_tensors="pt", padding=True)
    pixel_values = encoding.pixel_values
    
    text_inputs = processor.tokenizer(
        examples["text"],
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt"
    )
    labels = text_inputs.input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

# 14. Apply preprocessing with batched=True
processed_train_dataset = train_dataset.map(
    preprocess_data,
    batched=True,
    batch_size=4,
    remove_columns=train_dataset.column_names
)
processed_eval_dataset = eval_dataset.map(
    preprocess_data,
    batched=True,
    batch_size=4,
    remove_columns=eval_dataset.column_names
)

# 15. Set format for PyTorch
processed_train_dataset.set_format("torch")
processed_eval_dataset.set_format("torch")

# 16. Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# 17. Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_spanish_notary",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    logging_dir="./logs",
    eval_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    num_train_epochs=3,
    save_steps=100,
    eval_steps=100,
    logging_steps=50,
    learning_rate=3e-5,
    warmup_steps=50,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,
    predict_with_generate=True,
    report_to="none"
)

# 18. Create a custom data collator to ensure correct batching
def custom_data_collator(features):
    pixel_values_list = [f["pixel_values"] for f in features]
    labels_list = [f["labels"] for f in features]
    
    # Stack pixel_values and labels into batches
    pixel_values = torch.stack(pixel_values_list)
    labels = torch.stack(labels_list)
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

# 19. Initialize trainer with custom collator
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_eval_dataset,
    data_collator=custom_data_collator
)

# 20. Start training
print("🚀 Starting training...")
trainer.train()

# 21. Save the fine-tuned model
trainer.save_model("./trocr_spanish_notary_final")
print("✅ Training completed and model saved!")
