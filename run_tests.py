"""
Test script to run REBEL model on test files and output results to a file.
Outputs sentence and extracted relations in CSV format.
"""

import csv
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def extract_triplets(text):
    """Extract triplets from model output text with improved sensitivity."""
    triplets = []
    relation = ''
    subject = ''
    object_ = ''
    current = None
    
    # Clean the text first
    text = text.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
    
    for token in text.split():
        if token == "<triplet>":
            current = 't'
            if relation.strip() != '' and subject.strip() != '' and object_.strip() != '':
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation.strip() != '' and subject.strip() != '' and object_.strip() != '':
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    
    # Don't forget the last triplet
    if subject.strip() != '' and relation.strip() != '' and object_.strip() != '':
        triplets.append((subject.strip(), relation.strip(), object_.strip()))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_triplets = []
    for triplet in triplets:
        triplet_key = (triplet[0].lower(), triplet[1].lower(), triplet[2].lower())
        if triplet_key not in seen:
            seen.add(triplet_key)
            unique_triplets.append(triplet)
    
    return unique_triplets

def load_model():
    """Load the REBEL model and tokenizer."""
    print("Loading REBEL model...")
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    
    if torch.cuda.is_available():
        model = model.to("cuda:0")
        print("Model loaded on GPU")
    else:
        print("Model loaded on CPU")
    
    model.eval()
    return tokenizer, model

def load_test_sentences(tests_dir="tests"):
    """Load all sentences from test files."""
    test_files = [
        # "cardiology_reports.txt",
        # "gastroenterology_reports.txt", 
        # "neurology_reports.txt",
        # "orthopedics_reports.txt",
        # "pulmonology_reports.txt",
        "urology_reports.txt",
        "cholelithiasis_reports.txt",
        # "test_sentences.txt",
        # "test_cases.txt"
    ]
    
    all_sentences = []
    
    for filename in test_files:
        filepath = os.path.join(tests_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
                for sentence in sentences:
                    all_sentences.append((sentence, filename.replace('.txt', '')))
    
    return all_sentences

def is_location_relation(relation):
    """Check if a relation is exactly 'location'."""
    return relation.lower().strip() == 'location'

def show_all_relations(relation):
    """Accept all relations for debugging purposes."""
    return True

def run_tests_and_save_results(output_file="test_results.csv"):
    """Run tests on all test files and save results to CSV file."""
    
    # Load model
    tokenizer, model = load_model()
    
    # Load test sentences
    test_sentences = load_test_sentences()
    
    # Generation parameters - More comprehensive relation extraction
    gen_kwargs = {
        "max_length": 512,  # Longer outputs for more relations
        "length_penalty": 0,  # Neutral length penalty
        "num_beams": 5,  # More beam search paths
        "num_return_sequences": 3,  # Multiple sequences for more coverage
        "do_sample": False,  # Deterministic output
    }
    
    # Prepare results
    results = []
    
    print(f"Running tests on {len(test_sentences)} sentences...")
    
    for i, (sentence, source_file) in enumerate(test_sentences):
        print(f"Processing {i+1}/{len(test_sentences)}: {sentence[:50]}...")
        
        # Tokenize input
        model_inputs = tokenizer(sentence, max_length=512, padding=True, truncation=True, return_tensors='pt')
        
        # Generate predictions
        generated_tokens = model.generate(
            model_inputs["input_ids"].to(model.device),
            attention_mask=model_inputs["attention_mask"].to(model.device),
            **gen_kwargs,
        )
        
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        decoded_preds = [pred.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip() for pred in decoded_preds]
        
        # Extract triplets from each prediction
        found_relations = False
        all_triplets = []
        
        for pred_idx, prediction in enumerate(decoded_preds):
            if prediction and len(prediction) > 5:  # Only process meaningful predictions
                triplets = extract_triplets(prediction)
                all_triplets.extend(triplets)
                
                if triplets:
                    for subject, relation, object_ in triplets:
                        # Show all relations (no filtering)
                        found_relations = True
                        # Format: subject -> relation -> object
                        relation_str = f"{subject} -> {relation} -> {object_}"
                        results.append([sentence, relation_str, source_file])
        
        # If no triplets found in any prediction, still record the sentence
        if not found_relations:
            results.append([sentence, "No relations found", source_file])
    
    # Save results to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sentence', 'Relation', 'Source_File'])  # Header
        writer.writerows(results)
    
    print(f"Results saved to {output_file}")
    print(f"Total rows written: {len(results)}")
    
    # Also save a summary
    summary_file = output_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("REBEL Model Test Results Summary - All Relations\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Total sentences processed: {len(test_sentences)}\n")
        
        all_relations = [r for r in results if r[1] != 'No relations found']
        no_relations = [r for r in results if r[1] == 'No relations found']
        
        f.write(f"Relations extracted: {len(all_relations)}\n")
        f.write(f"Sentences with no relations: {len(no_relations)}\n\n")
        
        # Summary by source file
        from collections import Counter
        source_counts = Counter([r[2] for r in all_relations])
        f.write("Relations by source file:\n")
        f.write("-" * 25 + "\n")
        for source, count in source_counts.items():
            f.write(f"{source}: {count} relations\n")
        
        # Count relation types
        relation_types = Counter()
        for sentence, relation, source in all_relations:
            if " -> " in relation:
                rel_type = relation.split(" -> ")[1]  # Get the middle part (relation type)
                relation_types[rel_type] += 1
        
        f.write(f"\nRelation types found:\n")
        f.write("-" * 20 + "\n")
        for rel_type, count in relation_types.most_common():
            f.write(f"{rel_type}: {count} occurrences\n")
        
        f.write("\nSample Relations:\n")
        f.write("-" * 15 + "\n")
        for i, (sentence, relation, source) in enumerate(all_relations[:20]):  # Show first 20 results
            f.write(f"{i+1}. [{source}] {sentence}\n")
            f.write(f"   -> {relation}\n\n")
    
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    run_tests_and_save_results("rebel_test_results.csv")
