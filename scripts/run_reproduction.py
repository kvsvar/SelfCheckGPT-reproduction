import numpy as np
import yaml
import json
import os
import argparse
from sklearn.metrics import average_precision_score
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-inference', action='store_true', help='Skip local inference (use Kaggle outputs)')
    args = parser.parse_args()
    
    # Load config
    with open('configs/reproduction.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    targets = config['targets']
    
    # Load scores
    try:
        sentence_scores = np.load('outputs/scores/sentence_scores.npy')
    except Exception as e:
        print("Error loading sentence_scores.npy:", e)
        print("Make sure you copied Kaggle outputs to outputs/scores/")
        return
        
    print(f"Loaded sentence scores of shape: {sentence_scores.shape}")
    
    # Load dataset
    print("Loading dataset for labels...")
    dataset = load_dataset(config['dataset']['name'], split=config['dataset']['split'])
    
    # Extract labels
    labels = []
    for passage in dataset:
        for annotation in passage['annotation']:
            labels.append(annotation)
            
    if len(labels) != len(sentence_scores):
        print(f"Error: label count ({len(labels)}) != score count ({len(sentence_scores)})")
        return
        
    # Map labels for Nonfact task
    # nonfact: accurate->0, minor->1, major->1
    nonfact_y = []
    for l in labels:
        if l == 'accurate':
            nonfact_y.append(0)
        else:
            nonfact_y.append(1)
            
    # Map labels for Factual task
    # factual: accurate->1, minor->0, major->0
    factual_y = []
    for l in labels:
        if l == 'accurate':
            factual_y.append(1)
        else:
            factual_y.append(0)
            
    # Calculate AUC-PR
    # Nonfact task: predict 1 for hallucinated. Our score is P(contra) which means hallucinated.
    nonfact_aucpr = average_precision_score(nonfact_y, sentence_scores) * 100
    
    # Factual task: predict 1 for factual. Our score is P(contra) (hallucinated).
    # To predict factual, we invert it by taking -score.
    factual_aucpr = average_precision_score(factual_y, -sentence_scores) * 100
    
    # Calculate Random Baselines
    nonfact_random = (sum(nonfact_y) / len(nonfact_y)) * 100
    factual_random = (sum(factual_y) / len(factual_y)) * 100
    
    # Print report
    print("\n" + "="*50)
    print("FINAL REPRODUCTION REPORT")
    print("="*50)
    print(f"{'Task':<20} | {'Target':<10} | {'Achieved':<10}")
    print("-" * 45)
    print(f"{'Non-Factual AUC-PR':<20} | {targets['nonfact_aucpr']:<10.2f} | {nonfact_aucpr:<10.2f}")
    print(f"{'Factual AUC-PR':<20} | {targets['factual_aucpr']:<10.2f} | {factual_aucpr:<10.2f}")
    print("-" * 45)
    print(f"Baseline (Random)   | NF: {targets['random_nonfact_aucpr']:<5.2f} | {nonfact_random:<5.2f}")
    print(f"Baseline (Random)   | FA: {targets['random_factual_aucpr']:<5.2f} | {factual_random:<5.2f}")
    print("="*50 + "\n")
    
    # Save to metrics
    os.makedirs('outputs/metrics', exist_ok=True)
    results = {
        'nonfact_aucpr': nonfact_aucpr,
        'factual_aucpr': factual_aucpr,
        'target_nonfact': targets['nonfact_aucpr'],
        'target_factual': targets['factual_aucpr']
    }
    with open('outputs/metrics/reproduction_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
