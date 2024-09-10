import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
import sys
torch.cuda.set_per_process_memory_fraction(0.9)

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("CUDA is not available. Exiting training.")
    torch.cuda.set_per_process_memory_fraction(0.9)
    sys.exit()

print(f"Using device: {device}")

# Load dataset
dataset = load_dataset('xsum', split='train[:10%]')

# Initialize tokenizer and model
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)  # Move model to GPU if available

# Prepare data
def preprocess_function(examples):
    inputs = examples['document']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(examples['summary'], max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",  # Corrected to 'evaluation_strategy'
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',  # Add logging directory
    logging_steps=10,      # Log every 10 steps
)

# Use a data collator for dynamic padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model_path = './fine-tuned-t5-model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)