import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
This module loads data from a specified directory and plots the scores.
The scores include Bleu and Rouge1, Rouge2, RougeL, RougeLsum.
"""

def load_data_from_directory(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                for result in json_data.get("results", []):
                    data.append({
                        "model_name": json_data.get("model_name"),
                        "model_size": json_data.get("model_size"),  # Add model size
                        "use_ft": json_data.get("use_ft"),
                        "use_rag": json_data.get("use_rag"),
                        "bleu_score": result.get("bleu_score", None),
                        "rouge1": result.get('rouge_scores', {}).get('rouge1', None),
                        "rouge2": result.get('rouge_scores', {}).get('rouge2', None),
                        "rougeL": result.get('rouge_scores', {}).get('rougeL', None),
                        "rougeLsum": result.get('rouge_scores', {}).get('rougeLsum', None)
                    })
    return data

def plot_scores(df, score_type):
    plt.figure(figsize=(14, 8))
    
    # Order models by size
    model_order = df.sort_values(['model_size', 'model_name'])['model_name'].unique()

    split_n = lambda n, model_name: [model_name[i:i+n] for i in range(0, len(model_name), n)]
    xticks = ["\n".join(wrap_label) for wrap_label in [split_n(10, model_name) for model_name in model_order]]

    # Create a boxplot with `model_name` as the x-axis and `variant` as hue
    sns.boxplot(x='model_name', y=score_type, hue='variant', data=df, order=model_order)
    
    # Customize the plot to show model labels only once
    plt.title(f'{score_type.replace("_", " ").title()} by Model Variants (FT and RAG), Ordered by Size')
    plt.xlabel('Model Name')
    plt.ylabel(score_type.replace("_", " ").title())
    plt.ylim(0, 1)  # Set the y-axis range from 0 to 1
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title='Variants', loc='upper right', bbox_to_anchor=(1, 1))
    plt.xticks(range(14), xticks, rotation=0, ha='center')
    plt.tight_layout()
    plt.show()

# Load data from the specified directory
directory_path = '../eval_results'
data = load_data_from_directory(directory_path)

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Create a new column to represent the combination of use_ft and use_rag
df['variant'] = df.apply(lambda row: f"FT: {row['use_ft']}, RAG: {row['use_rag']}", axis=1)

# Plot the scores
plot_scores(df, "rouge1")
plot_scores(df, "bleu_score")
plot_scores(df, "rouge2")
plot_scores(df, "rougeL")
plot_scores(df, "rougeLsum")