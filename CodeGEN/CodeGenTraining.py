import torch
import matplotlib.pyplot as plt
from datasets import load_metric
import pandas as pd
import time
import re
import ast
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM,PreTrainedTokenizer
import torch
#import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import argparse
from datasets import load_dataset, dataset_dict, DatasetDict, load_metric
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import T5Tokenizer,AutoTokenizer, RobertaTokenizer,CodeGenTokenizerFast,CodeGenModel,CodeGenForCausalLM
import torch
from torch.utils.data import Dataset, DataLoader
import json
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine
import numpy as np
from Levenshtein import distance as levenshtein_distance
import sys

#original_stdout = sys.stdout   
#log_file = open("/home/biradar/Desktop/NLPBasedTCGeneration/CodeGen/results/logs.txt", "w")
#sys.stdout = log_file
# Load the dataset
data = pd.read_csv('../Final_Data_Generated.csv', quoting=3, on_bad_lines='skip')
#data.head()


# Split data into train and test sets (60% train, 20% validation, 20% test)

# Step 1: Split the dataset into training + validation and testing sets
df_train_val, df_test = train_test_split(data, test_size=0.2, random_state=42)

# Step 2: Split the training + validation set into training and validation sets
df_train, df_val = train_test_split(df_train_val, test_size=0.25, random_state=42)

# Optionally, you can save the splits to CSV files
df_train.to_csv('data/train_data.csv', index=False)
df_val.to_csv('data/val_data.csv', index=False)
df_test.to_csv('data/test_data.csv', index=False)



# Dataset Class for PyTorch
class CodeGenDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer # Make sure the tokenizer variable is not overwritten in the global scope
        self.max_length = max_length

    def __len__(self):
        return len(self.data['Python_Function'])

    def __getitem__(self, idx):
        function =self.data.iloc[idx]['Python_Function']
        description = self.data.iloc[idx]['Description']
        unit_test_case = self.data.iloc[idx]['Unit Test Cases']
        
        additional_prompt="Write unit tests for this function, ensuring to include assertions for expected outputs."
        input_text = str(f"Function: {function} Description: {additional_prompt}{description}")
        target_text = str(unit_test_case)

        # Tokenize input and output
        input_encoding = self.tokenizer(input_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        target_encoding = self.tokenizer(target_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        # Return input_ids and labels
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten(),
        }
    
# Tokenizer and Model Initialization
#tokenizer = CodeGenTokenizerFast.from_pretrained("Salesforce/codegen-2B-mono")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi") 
#print("tokenizer config->",tokenizer.init_kwargs)
#print("Pad token:", tokenizer.pad_token)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "left"
model = CodeGenForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")

# Create PyTorch Dataset and DataLoader
train_dataset = CodeGenDataset(df_train, tokenizer)
val_dataset = CodeGenDataset(df_val, tokenizer)
test_dataset = CodeGenDataset(df_test, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=4)
val_dataloader = DataLoader(val_dataset, batch_size=4)
test_dataloader = DataLoader(test_dataset, batch_size=4)
print("dataset loaded")


# Function to generate predictions and calculate validation loss
def generate_predictions_and_loss(model, dataloader, tokenizer, device, max_length=1024):
    model.eval()
    predictions = []
    references = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Calculate loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Generate predictions
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)

            # Decode the generated and reference sequences
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            refs = [tokenizer.decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=True) for l in labels]

            predictions.extend(preds)
            references.extend(refs)

    # Calculate average validation loss
    avg_val_loss = total_loss / len(dataloader)
    # Save predictions and references
    with open("predictions.json", "w") as f:
        json.dump(predictions, f)
    with open("references.json", "w") as f:
        json.dump(references, f)

    return predictions, references, avg_val_loss


# Load BLEU and ROUGE metrics
bleu_metric = load_metric("sacrebleu")
rouge_metric = load_metric("rouge")

# Function to evaluate exact match accuracy
def exact_match(predictions, references):
    exact_matches = sum([1 if pred == ref else 0 for pred, ref in zip(predictions, references)])
    return exact_matches / len(references)

# Function to calculate BLEU score
def compute_bleu(predictions, references):
    bleu = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    return bleu['score']/100

# Function to calculate ROUGE score
def compute_rouge(predictions, references):
    rouge = rouge_metric.compute(predictions=predictions, references=references)
    return rouge


def normalize_whitespace(code):
    """Normalize whitespace by removing extra spaces, newlines, and consistent indentation."""
    # Replace multiple whitespace with a single space
    code = re.sub(r'\s+', ' ', code)
    return code.strip()  # Remove leading and trailing whitespace

def parse_and_unparse_code(code):
    """Parse code into an AST and then unparse it back to code for standardized formatting."""
    try:
        # Parse code into an AST
        tree = ast.parse(code)
        # Convert back to code with standardized formatting
        return ast.unparse(tree)
    except SyntaxError:
        # If parsing fails, return the original code
        return code
    
def format_code(code):
    
    code = normalize_whitespace(code)
    code = parse_and_unparse_code(code)
    return code

# Function to tokenize code
def tokenize_code(code):
    """Tokenize the code into individual words/tokens."""
    return code.split()

# Function to calculate normalized Levenshtein similarity
def compute_levenshtein(predictions, references):
    """
    Computes the Levenshtein similarity for a list of predictions and references.
    Similarity is scaled between 0 and 1.
    """
    similarities = []
    for pred, ref in zip(predictions, references):
        lev_distance = levenshtein_distance(pred, ref)
        max_len = max(len(pred), len(ref))
        # Normalize Levenshtein distance to similarity (1 - normalized distance)
        similarity = 1 - (lev_distance / max_len) if max_len > 0 else 1
        similarities.append(similarity)
    # Aggregate similarities to get a single value per epoch
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0  # Compute mean
    return avg_similarity

# Function to calculate normalized Cosine similarity
def compute_cosine_similarity(predictions, references):
    """
    Computes the average cosine similarity between predictions and references.
    Similarity is scaled between 0 and 1.
    """
    vectorizer = CountVectorizer().fit(predictions + references)
    similarities = []
    
    for pred, ref in zip(predictions, references):
        # Transform into vector space
        pred_vector = vectorizer.transform([pred]).toarray().flatten()
        ref_vector = vectorizer.transform([ref]).toarray().flatten()
        # Calculate cosine similarity
        if np.all(pred_vector == 0) or np.all(ref_vector == 0):
            sim = 0  # Avoid division by zero for empty vectors
        else:
            sim = 1 - cosine(pred_vector, ref_vector) if not np.all(pred_vector == 0) and not np.all(ref_vector == 0) else 0
        
        similarities.append(sim)
    
    # Aggregate similarities to get a single value per epoch
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0  # Compute mean
    return avg_similarity

# Lists to store metrics for each batch and epoch
batch_train_losses = []
epoch_train_losses = []
epoch_val_losses = []
epoch_exact_match_scores = []
epoch_bleu_scores = []
epoch_rouge_l_scores = []
epoch_levenshtein_score = []
epoch_cosine_score = []
epoch_precisions = []
epoch_recalls = []
epoch_f1_scores = []
predicted_tokens = []
reference_tokens = []

#device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.cuda.empty_cache()

# Training
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 10
gradient_accumulation_steps = 2 
start_time = time.time()

# Training loop with metrics collection for all epochs
for epoch in range(num_epochs):
    # Initialize variables for epoch metrics
    total_train_loss = 0
    model.train()
    
    # Training phase
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,labels=labels)
        loss = outputs.loss

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

        # Track training loss for this batch
        batch_train_losses.append(loss.item())
        total_train_loss += loss.item()

    # Average training loss for this epoch
    avg_epoch_train_loss = total_train_loss / len(train_dataloader)
    epoch_train_losses.append(avg_epoch_train_loss)

    # Evaluation phase
    predictions, references, avg_val_loss = generate_predictions_and_loss(model, val_dataloader, tokenizer, device)
    epoch_val_losses.append(avg_val_loss)

    
    # Compute Levenshtein similarity
    levenshtein_score = compute_levenshtein(predictions, references)
    
    # Compute Cosine similarity
    cosine_score = compute_cosine_similarity(predictions, references)

    # Calculate evaluation metrics
    em_accuracy = exact_match(predictions, references)
    bleu_score = compute_bleu(predictions, references)
    rouge_scores = compute_rouge(predictions, references)
    rouge_l_score = rouge_scores['rougeL'].mid.fmeasure

    # Store epoch-level metrics
    epoch_exact_match_scores.append(em_accuracy)
    epoch_bleu_scores.append(bleu_score)
    epoch_rouge_l_scores.append(rouge_l_score)
    epoch_levenshtein_score.append(levenshtein_score)
    epoch_cosine_score.append(cosine_score)

    # Print metrics for each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Training Loss: {avg_epoch_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    print(f"Exact Match Accuracy: {em_accuracy:.4f}, BLEU Score: {bleu_score:.4f}, ROUGE-L Score: {rouge_l_score:.4f}\n")
    
    
    # Tokenize and binarize the aggregated predictions and references for the epoch
    mlb = MultiLabelBinarizer()
    predicted_tokens = [tokenize_code(pred) for pred in predictions]
    reference_tokens = [tokenize_code(ref) for ref in references]
    predictions_binary = mlb.fit_transform(predicted_tokens)
    references_binary = mlb.transform(reference_tokens)

    # Compute precision, recall, and F1 for the epoch
    precision = precision_score(references_binary, predictions_binary,average='micro')
    recall = recall_score(references_binary, predictions_binary,average='micro')
    f1 = f1_score(references_binary, predictions_binary,average='micro')

    # Store epoch-level metrics
    epoch_precisions.append(precision)
    epoch_recalls.append(recall)
    epoch_f1_scores.append(f1)

# Save the results to a CSV file after the training loop is complete
# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'epoch': range(1, num_epochs + 1),
    'train_loss': epoch_train_losses,
    'val_loss': epoch_val_losses,
    'levenshtein_score': epoch_levenshtein_score,
    'cosine_score' : epoch_cosine_score,
    'bleu_score': epoch_bleu_scores,
    'rouge_l_score': epoch_rouge_l_scores,
    'precision' : epoch_precisions,
    'recall' : epoch_recalls,
    'f1' : epoch_f1_scores
})

# Save the results to a CSV file
results_df.to_csv('results/training_results.csv', index=False)
print("Training results saved to training_results1.csv")

# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, int(num_epochs)+1), epoch_train_losses, label='Epoch Training Loss')
plt.plot(range(1, int(num_epochs)+1), epoch_val_losses, label='Epoch Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.show()

plt.savefig('/NLPBasedTCGeneration/CodeGen/results/training_metrics.png')
print("Training metrics plot saved to training_metrics.png") 

# Save the fine-tuned model
model.save_pretrained('results/codeGen_finetuned')
tokenizer.save_pretrained('results/codeGen_finetuned')

end_time = time.time()
print("total time taken: ", (end_time-start_time) / 60)

# Close the log file
#log_file.close()

# Reset standard output to console
#sys.stdout = original_stdout