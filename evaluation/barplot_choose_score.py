import os
import json
import matplotlib.pyplot as plt
import numpy as np

"""
This module loads data from a specified directory and plots the scores.
The scores include Bleu and Rouge1, Rouge2, RougeL, RougeLsum.
"""

def load_data(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                json_data = json.load(file)
                data.append(json_data)
    return data

def plot_scores(data, selected_score):
    # Extract unique model names
    models = sorted(list(set(item["model_name"] for item in data)),
                    key=lambda x: (next(d for d in data if d["model_name"] == x)["model_size"], x))

    # Find all unique configurations
    configurations = sorted(
        list(set((d["use_ft"], d["use_rag"]) for d in data)))
    n_configs = len(configurations)

    # Prepare labels for x-axis
    config_labels = [f"FT: {ft}, RAG: {rag}" for ft, rag in configurations]

    # Initialize score dictionary with None for missing values
    score_dict = {
        'Mean BLEU': {model: [None] * n_configs for model in models},
        'Mean ROUGE1': {model: [None] * n_configs for model in models},
        'Mean ROUGE2': {model: [None] * n_configs for model in models},
        'Mean ROUGEL': {model: [None] * n_configs for model in models},
        'Mean ROUGELsum': {model: [None] * n_configs for model in models}
    }

    for model in models:
        for d in data:
            if d["model_name"] == model:
                config_idx = configurations.index((d["use_ft"], d["use_rag"]))
                score_dict['Mean BLEU'][model][config_idx] = d["mean_bleu_score"]
                score_dict['Mean ROUGE1'][model][config_idx] = d["mean_rouge_scores"]["rouge1"]
                score_dict['Mean ROUGE2'][model][config_idx] = d["mean_rouge_scores"]["rouge2"]
                score_dict['Mean ROUGEL'][model][config_idx] = d["mean_rouge_scores"]["rougeL"]
                score_dict['Mean ROUGELsum'][model][config_idx] = d["mean_rouge_scores"]["rougeLsum"]

    x = np.arange(n_configs)
    total_width = 0.8  # Total width for all bars in a group
    width = total_width / len(models)  # Individual bar width

    # Plot the selected score type
    fig, ax = plt.subplots(figsize=(12, 6))

    scores = score_dict[selected_score]
    for i, model in enumerate(models):
        # Replace None with 0
        model_scores = [0 if score is None else score for score in scores[model]]
        ax.bar(x + i * width - total_width / 2, model_scores, width, label=model)
    
    ax.set_title(selected_score)
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=0, ha='right')
    ax.set_ylim(0, 1)  # Set the y-axis range from 0 to 1
    ax.grid(True)

    # Add a legend
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Load data from the specified directory
directory = "../eval_results/"
data = load_data(directory)

# Prompt user to select score type
score_types = ['Mean BLEU', 'Mean ROUGE1', 'Mean ROUGE2', 'Mean ROUGEL', 'Mean ROUGELsum']
print("Select a score type to plot:")
for i, score in enumerate(score_types, 1):
    print(f"{i}. {score}")

choice = int(input("Enter the number of your choice: "))
selected_score = score_types[choice - 1]

# Plot scores
plot_scores(data, selected_score)