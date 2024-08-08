import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
from matplotlib.ticker import FixedLocator

def load_json_files(file_pattern):
    files = glob.glob(file_pattern)
    data = []
    for file in files:
        with open(file, 'r') as f:
            data.append(json.load(f))
    return data

def plot_question_duration(data):
    # Extract relevant data into a DataFrame
    rows = []
    for entry in data:
        model_name = entry['model_name']
        use_ft = entry['use_ft']
        use_rag = entry['use_rag']
        model_size = entry.get('model_size', 'Unknown')
        for result in entry['results']:
            duration = result['duration_ms']
            rows.append({
                'model_name': model_name,
                'use_ft': use_ft,
                'use_rag': use_rag,
                'model_size': model_size,
                'duration_ms': duration
            })

    df = pd.DataFrame(rows)

    # Convert boolean columns to categorical for better plotting
    df['use_ft'] = df['use_ft'].astype('category')
    df['use_rag'] = df['use_rag'].astype('category')

    # Create a new column to represent the combination of 'use_ft' and 'use_rag'
    df['ft_rag_variant'] = 'FT: ' + df['use_ft'].astype(str) + ', RAG: ' + df['use_rag'].astype(str)

    # Sort by model_size and then model_name
    df = df.sort_values(by=['model_size', 'model_name'])

    # Wrap the model names to avoid rotation
    def wrap_labels(ax, width):
        labels = []
        for label in ax.get_xticklabels():
            text = label.get_text()
            wrapped_text = "\n".join(textwrap.wrap(text, width))
            labels.append(wrapped_text)
        ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
        ax.set_xticklabels(labels, rotation=0)

    # Plotting
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x='model_name', y='duration_ms', hue='ft_rag_variant', data=df)
    plt.xlabel('Model Name')
    plt.ylabel('Duration (ms)')
    plt.title('Question Duration Distribution with RAG and FT Variations')
    plt.legend(title='Variants', loc='upper right')

    # Wrap the x labels
    wrap_labels(ax, 10)

    plt.tight_layout()
    plt.show()

# Usage
data = load_json_files("../eval_results/*.json")
plot_question_duration(data)