import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                json_data = json.load(file)
                data.append((filename, json_data))
    return data

def plot_scores(data):
    metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    metric_labels = {
        'bleu': 'Mean BLEU Score',
        'rouge1': 'Mean ROUGE-1 Score',
        'rouge2': 'Mean ROUGE-2 Score',
        'rougeL': 'Mean ROUGE-L Score',
        'rougeLsum': 'Mean ROUGE-LSum Score'
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    # Iterate over metrics and create the subplots
    for idx, metric in enumerate(metrics):
        model_sizes = []
        scores = []
        avg_power_watts = []
        model_names = []
        ft_rag_labels = []

        # Iterate through all loaded JSON data
        for filename, entry in data:
            model_size = entry.get('model_size')
            avg_power_watt = entry.get('mean_avg_power', None)
            use_ft = entry.get('use_ft')
            use_rag = entry.get('use_rag')
            model_name = entry.get('model_name')

            if metric == 'bleu':
                score = entry.get('mean_bleu_score')
            else:
                score = entry.get('mean_rouge_scores', {}).get(metric)

            if model_size and score is not None:
                model_sizes.append(float(model_size.replace('B', '')))
                scores.append(score)
                avg_power_watts.append(avg_power_watt if avg_power_watt is not None else 50)  # Default size if not found
                model_names.append(model_name)
                ft_rag_labels.append(f'FT: {"✓" if use_ft else "X"}, RAG: {"✓" if use_rag else "X"}')

        # Sort the data by model sizes to ensure proper trendline plotting
        sorted_indices = np.argsort(model_sizes)
        model_sizes = np.array(model_sizes)[sorted_indices]
        scores = np.array(scores)[sorted_indices]

        # Assign a unique color to each model name
        unique_models = list(set(model_names))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_models)))
        model_color_map = dict(zip(unique_models, colors))

        # Create scatter plot
        ax = axes[idx]
        for i, model_name in enumerate(model_names):
            ax.scatter(model_sizes[i], scores[i], s=avg_power_watts[i], color=model_color_map[model_name], alpha=0.5, label=model_name if i == model_names.index(model_name) else "")

        # Add labels for FT and RAG values
        for i, label in enumerate(ft_rag_labels):
            ax.text(model_sizes[i], scores[i], label, fontsize=7, ha='center', va='bottom')

        # Fit and plot the trendline
        z = np.polyfit(model_sizes, scores, 1)
        p = np.poly1d(z)
        ax.plot(model_sizes, p(model_sizes), "r--", label="Trendline")

        ax.set_xlabel('Model Size (Billion Parameters)')
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(f'Model Size by {metric_labels[metric]} (Point Size by Average Watt)')
        ax.grid(True)

    # Hide any unused subplots
    for i in range(len(metrics), len(axes)):
        fig.delaxes(axes[i])

    # Create legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), title="Model Names", loc='lower right')

    plt.tight_layout()
    plt.show()

# Directory containing the evaluation result JSON files
eval_results_dir = '../eval_results'
# Load data from the specified directory
data = load_data(eval_results_dir)
# Plot scores
plot_scores(data)