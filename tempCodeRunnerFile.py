import pyarrow as pa
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Define a custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Example URL for downloading the dataset
dataset_url = "https://datasets-server.huggingface.co/rows?dataset=jpwahle%2Fmachine-paraphrase-dataset&config=default&split=train&offset=0&length=100"

# Download the dataset
response = requests.get(dataset_url)
data = response.json()

# Extract features and rows
rows = data['rows']

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Process data
texts = [row['row']['text'] for row in rows]
labels = [row['row']['label'] for row in rows]

# Tokenize the texts
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Create dataset
dataset = CustomDataset(encodings, labels)

# Initialize model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./fine-tuned-model')
