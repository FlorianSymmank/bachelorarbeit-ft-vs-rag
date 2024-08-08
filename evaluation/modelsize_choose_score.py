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

def plot_scores(data, metric='bleu'):
    # Initialize lists to store model sizes, scores, average power wattage, and labels
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
            # Default size if not found
            avg_power_watts.append(avg_power_watt**1.25 if avg_power_watt is not None else 50)
            model_names.append(model_name)
            ft_rag_labels.append(f'FT:{"✓" if use_ft else "X"}, RAG:{"✓" if use_rag else "X"}')

    # Assign a unique color to each model name
    unique_models = list(set(model_names))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_models)))
    model_color_map = dict(zip(unique_models, colors))

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    for i, model_name in enumerate(model_names):
        plt.scatter(model_sizes[i], scores[i], s=avg_power_watts[i], color=model_color_map[model_name],
                    alpha=0.5, label=model_name if i == model_names.index(model_name) else "")

    # Add labels for FT and RAG values
    for i, label in enumerate(ft_rag_labels):
        plt.text(model_sizes[i], scores[i], label, fontsize=9, ha='center', va='bottom')

    # Fit a linear trendline to the data
    z = np.polyfit(model_sizes, scores, 1)
    p = np.poly1d(z)
    plt.plot(model_sizes, p(model_sizes), "r--", label="Trendline")

    # Create legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Model Names", loc='lower right')

    plt.xlabel('Model Size (Billion Parameters)')
    plt.ylabel(f'Mean {metric.upper()} Score')
    plt.title(f'Model Size by Mean {metric.upper()} Score (Point Size by Average Watt)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Directory containing the evaluation result JSON files
eval_results_dir = '../eval_results'
# Load data from the specified directory
data = load_data(eval_results_dir)
# Plot scores with BLEU metric
plot_scores(data, metric='bleu')
# To plot scores with ROUGE1 metric, use:
plot_scores(data, metric='rouge1')
# To plot scores with ROUGE2 metric, use:
plot_scores(data, metric='rouge2')
# To plot scores with ROUGEL metric, use:
plot_scores(data, metric='rougeL')
# To plot scores with ROUGELSUM metric, use:
plot_scores(data, metric='rougeLsum')