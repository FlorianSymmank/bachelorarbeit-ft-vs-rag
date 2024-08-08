import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator

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

def wrap_labels(ax, width):
    labels = ax.get_xticklabels()
    wrapped_labels = ['\n'.join(label.get_text()[i:i+width] for i in range(0, len(label.get_text()), width)) for label in labels]
    ax.set_xticklabels(wrapped_labels, ha='center')

def plot_scores(df, score_type=None):
    scores = ['bleu_score', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    if score_type:
        scores = [score_type]
    
    num_scores = len(scores)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 18))
    axes = axes.flatten()
    
    handles, labels = None, None

    # Order models by size
    model_order = df.sort_values(['model_size', 'model_name'])['model_name'].unique()
    
    for ax, score in zip(axes, scores):
        sns.boxplot(ax=ax, x='model_name', y=score, hue='variant', data=df, order=model_order)
        ax.set_title(f'{score.split("_")[0].title()} Score Distribution by Model and Configuration')
        ax.set_xlabel('')
        ax.set_ylabel(score.replace("_", " ").title())
        if handles is None and labels is None:
            handles, labels = ax.get_legend_handles_labels()
        ax.set_ylim(0, 1)
        ax.legend().remove()
        ax.tick_params(axis='x', labelright=True)
        ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))  # Set the ticks explicitly
        wrap_labels(ax, 10)  # Adjust the width as necessary
    
    # Hide any unused subplots
    for i in range(num_scores, len(axes)):
        fig.delaxes(axes[i])
    
    # Add a single legend below all plots
    fig.legend(handles=handles, labels=labels, title='Variants', loc='lower right', ncol=1)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust the layout to make space for the legend
    plt.show()

# Load data from the specified directory
directory_path = '../eval_results'
data = load_data_from_directory(directory_path)

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Create a new column to represent the combination of use_ft and use_rag
df['variant'] = df.apply(lambda row: f"FT: {row['use_ft']}, RAG: {row['use_rag']}", axis=1)

# Sort DataFrame by model size (assuming model sizes are comparable as strings)
df = df.sort_values(by='model_size')

# Define the score type to plot (set to None to plot all scores)
score_type = None  # Change this to 'bleu_score', 'rouge1', 'rouge2', etc., or set to None to plot all

# Plot the scores
plot_scores(df, score_type)