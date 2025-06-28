import pandas as pd
import time
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer,AutoTokenizer, CodeGenTokenizerFast, CodeGenForCausalLM
import torch
#import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import argparse
from datasets import load_dataset, dataset_dict, DatasetDict
import matplotlib.pyplot as plt
from datasets import load_metric
import time
import re
import ast
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine
import numpy as np
from Levenshtein import distance as levenshtein_distance

import torch
from datasets import load_metric
from transformers import T5Tokenizer, T5ForConditionalGeneration,AutoModelForCausalLM


dataset = pd.read_csv("data/test_data.csv")

# Load the fine-tuned model and tokenizer
model_path = 'results/codeGen_finetuned'  # Path to the saved model
model = CodeGenForCausalLM.from_pretrained(model_path)
tokenizer = CodeGenTokenizerFast.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()


# Move the model to the appropriate device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load BLEU and ROUGE metrics
bleu_metric = load_metric("sacrebleu")
rouge_metric = load_metric("rouge")

# Function to generate predictions
def generate_unit_tests(python_function, description, model, tokenizer, device, max_length=512):
    # Prepare the input text by combining the function and description
    additional_prompt="Write unit tests for this function, ensuring to include assertions for expected outputs."
    input_text = f"Function: {python_function} Description: {additional_prompt}{description}"
    
    # Tokenize the input text
    input_encoding = tokenizer(input_text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    
    # Move input to the device (CPU/GPU)
    input_ids = input_encoding['input_ids'].to(device)
    attention_mask = input_encoding['attention_mask'].to(device)
    
    start_time = time.time()
    # Generate predictions (unit test cases) from the model
    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=50)
    
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    # Decode the generated tokens to text
    predicted_unit_tests = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return predicted_unit_tests,elapsed_time

# Perform inference and evaluation
# Initialize the ROUGE scorer
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

predictions = []
references = []
predictions_tokens = []
references_tokens = []
exact_match_scores = []
exact_match_accuracy = []
bleu_scores = []
levenshtein_scores = []
cosine_scores = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
sample_precisions = []
sample_recalls = []
sample_f1_scores = []
testcase_time = []

start_time = time.time()

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
    return similarities

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
    
    return similarities

for  _, example in dataset.iterrows():
    python_function = example['python_function']
    description = example['Description']
    reference_unit_test = example['groundtruth_unit_test'].strip()

    # Generate predicted unit test cases
    predicted_unit_test,elapsed_time = generate_unit_tests(python_function, description, model, tokenizer, device)
    
    # Apply formatting 
    predicted_unit_test = format_code(predicted_unit_test)
    reference_unit_test = format_code(reference_unit_test)

    # Store the predictions and references
    predictions.append(predicted_unit_test.strip())
    references.append(reference_unit_test)

    # Compute evaluation metrics for each sample
    # Exact Match Accuracy
    #exact_match_accuracy = [1 if pred.strip() == ref.strip() else 0 for pred, ref in zip(predictions, references)]
    #exact_match_accuracy.append(exact_matches)

    #exact_match = 1 if predicted_unit_test == reference_unit_test else 0
    #exact_match_scores.append(exact_match)

    # Compute BLEU score
    #bleu = sentence_bleu([reference_unit_test.split()], predicted_unit_test.split())
    bleu = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    bleu_scores.append(bleu['score']/100)
    
    # Compute Levenshtein similarity
    levenshtein_score = compute_levenshtein(predictions, references)
    
    # Compute Cosine similarity
    cosine_score = compute_cosine_similarity(predictions, references)

    # Compute ROUGE scores
    rouge = rouge_scorer_obj.score(reference_unit_test, predicted_unit_test)
    rouge1_scores.append(rouge['rouge1'].fmeasure)
    rouge2_scores.append(rouge['rouge2'].fmeasure)
    rougeL_scores.append(rouge['rougeL'].fmeasure)

    # Apply formatting to both predicted and reference unit tests
    predicted_unit_test = format_code(predicted_unit_test)
    reference_unit_test = format_code(reference_unit_test)

    # Store the formatted predictions and references
    # predictions_tokens.append(tokenize_code(predicted_unit_test))
    # references_tokens.append(tokenize_code(reference_unit_test))

    # Tokenize the formatted predictions and references
    predicted_tokens = tokenize_code(predicted_unit_test)
    reference_tokens = tokenize_code(reference_unit_test)

    # Convert tokens to binary format for the current sample
    mlb = MultiLabelBinarizer()
    sample_predictions_binary = mlb.fit_transform([predicted_tokens])[0]
    sample_references_binary = mlb.transform([reference_tokens])[0]

    # Compute precision, recall, and F1 for the current sample
    precision = precision_score(sample_references_binary, sample_predictions_binary)
    recall = recall_score(sample_references_binary, sample_predictions_binary)
    f1 = f1_score(sample_references_binary, sample_predictions_binary)

    # Store individual sample scores
    sample_precisions.append(precision)
    sample_recalls.append(recall)
    sample_f1_scores.append(f1)
    
    testcase_time.append(elapsed_time)

end_time = time.time()
print("Time required to generate test case:", (end_time-start_time) / 60)


# Display overall evaluation results
#print(f"Average Exact Match Accuracy: {sum(exact_match_scores) / len(exact_match_scores):.4f}")
print(f"Average BLEU Score: {sum(bleu_scores) / len(bleu_scores):.4f}")
print(f"Average ROUGE Scores: R1: {sum(rouge1_scores) / len(rouge1_scores):.4f}, "
      f"R2: {sum(rouge2_scores) / len(rouge2_scores):.4f}, RL: {sum(rougeL_scores) / len(rougeL_scores):.4f}")


# Create a DataFrame with the results
results_df = pd.DataFrame({
    'python_function': dataset['python_function'],
    'description': dataset['Description'],
    'predicted_unit_test': predictions,
    'groundtruth_unit_test': references,
    'levenshtein_score': levenshtein_score,
    'cosine_score' : cosine_score,
    'bleu_score': bleu_scores,
    'rouge1_score': rouge1_scores,
    'rouge2_score': rouge2_scores,
    'rougeL_score': rougeL_scores,
    'precision' : sample_precisions,
    'recall' : sample_recalls,
    'f1' : sample_f1_scores,
    'testcase generation time': testcase_time
})


# Save the DataFrame to a CSV file
results_csv = 'results/predicted_unit_tests.csv'
results_df.to_csv(results_csv, index=False)

print(f"Predicted unit test cases saved to {results_csv}")
