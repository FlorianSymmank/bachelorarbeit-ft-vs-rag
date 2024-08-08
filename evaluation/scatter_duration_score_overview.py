import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Directory containing the evaluation result files
eval_results_dir = '../eval_results'

def read_json_files(directory):
    """Read JSON files from a given directory."""
    data_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                data_list.append(data)
    return data_list

def parse_model_size(model_size_str):
    """Convert model size string to a numerical value."""
    try:
        # Remove the 'B' and convert to float, multiply by a factor for better visualization
        return float(model_size_str.replace('B', '')) * 7
    except ValueError:
        return 100  # Default size if parsing fails

def extract_data(data_list):
    """Extract relevant data from the JSON files."""
    durations = []
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    rougeLsum_scores = []
    model_sizes = []
    model_names = []
    rag_settings = []
    ft_settings = []

    for data in data_list:
        model_size_str = data.get('model_size', '1B')
        model_size = parse_model_size(model_size_str)  # Convert to numerical value
        model_name = data.get('model_name', 'Unknown')
        rag_setting = data.get('use_rag', False)
        ft_setting = data.get('use_ft', False)
        for result in data['results']:
            durations.append(result['duration_ms'])
            bleu_scores.append(result['bleu_score'])
            rouge1_scores.append(result['rouge_scores']['rouge1'])
            rouge2_scores.append(result['rouge_scores']['rouge2'])
            rougeL_scores.append(result['rouge_scores']['rougeL'])
            rougeLsum_scores.append(result['rouge_scores']['rougeLsum'])
            model_sizes.append(model_size)
            model_names.append(model_name)
            rag_settings.append(rag_setting)
            ft_settings.append(ft_setting)
    
    return durations, bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, rougeLsum_scores, model_sizes, model_names, rag_settings, ft_settings

def get_color_map(model_names):
    """Generate a color map based on unique model names."""
    unique_model_names = list(set(model_names))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_model_names)))
    color_map = {model_name: color for model_name, color in zip(unique_model_names, colors)}
    return color_map, unique_model_names

def get_marker(rag, ft):
    """Return marker style based on RAG and FT settings."""
    if rag and ft:
        return 'o'  # Circle
    elif rag and not ft:
        return 's'  # Square
    elif not rag and ft:
        return '^'  # Triangle
    else:
        return 'D'  # Diamond

def plot_data(durations, bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, rougeLsum_scores, model_sizes, model_names, rag_settings, ft_settings):
    """Plot the extracted data."""
    color_map, unique_model_names = get_color_map(model_names)
    colors = [color_map[model_name] for model_name in model_names]

    plt.figure(figsize=(15, 8))

    for rag, ft in [(True, True), (True, False), (False, True), (False, False)]:
        indices = [i for i in range(len(rag_settings)) if rag_settings[i] == rag and ft_settings[i] == ft]
        sub_durations = [durations[i] for i in indices]
        sub_bleu_scores = [bleu_scores[i] for i in indices]
        sub_rouge1_scores = [rouge1_scores[i] for i in indices]
        sub_rouge2_scores = [rouge2_scores[i] for i in indices]
        sub_rougeL_scores = [rougeL_scores[i] for i in indices]
        sub_rougeLsum_scores = [rougeLsum_scores[i] for i in indices]
        sub_model_sizes = [model_sizes[i] for i in indices]
        sub_colors = [colors[i] for i in indices]
        marker = get_marker(rag, ft)

        plt.subplot(2, 3, 1)
        plt.scatter(sub_durations, sub_bleu_scores, s=sub_model_sizes, c=sub_colors, marker=marker)
        plt.title('Question Time vs BLEU Score (Point Size by Model Size)')
        plt.xlabel('Duration (ms)')
        plt.ylabel('BLEU Score')

        plt.subplot(2, 3, 2)
        plt.scatter(sub_durations, sub_rouge1_scores, s=sub_model_sizes, c=sub_colors, marker=marker)
        plt.title('Question Time vs ROUGE-1 Score (Point Size by Model Size)')
        plt.xlabel('Duration (ms)')
        plt.ylabel('ROUGE-1 Score')

        plt.subplot(2, 3, 3)
        plt.scatter(sub_durations, sub_rouge2_scores, s=sub_model_sizes, c=sub_colors, marker=marker)
        plt.title('Question Time vs ROUGE-2 Score (Point Size by Model Size)')
        plt.xlabel('Duration (ms)')
        plt.ylabel('ROUGE-2 Score')

        plt.subplot(2, 3, 4)
        plt.scatter(sub_durations, sub_rougeL_scores, s=sub_model_sizes, c=sub_colors, marker=marker)
        plt.title('Question Time vs ROUGE-L Score (Point Size by Model Size)')
        plt.xlabel('Duration (ms)')
        plt.ylabel('ROUGE-L Score')

        plt.subplot(2, 3, 5)
        plt.scatter(sub_durations, sub_rougeLsum_scores, s=sub_model_sizes, c=sub_colors, marker=marker)
        plt.title('Question Time vs ROUGE-Lsum Score (Point Size by Model Size)')
        plt.xlabel('Duration (ms)')
        plt.ylabel('ROUGE-Lsum Score')

    # Create custom legend for colors
    color_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[model_name], markersize=10) for model_name in unique_model_names]
    # Create custom legend for markers
    marker_labels = ['RAG & FT', 'RAG', 'FT', 'Base']
    marker_styles = ['o', 's', '^', 'D']
    marker_handles = [plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray', markersize=10) for marker in marker_styles]

    # Positioning both legends side by side
    plt.figlegend(color_handles + marker_handles, unique_model_names + marker_labels, loc='lower right', bbox_to_anchor=(0.85, 0.1), ncol=1)

    plt.tight_layout()
    plt.show()

def main():
    data_list = read_json_files(eval_results_dir)
    durations, bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, rougeLsum_scores, model_sizes, model_names, rag_settings, ft_settings = extract_data(data_list)
    plot_data(durations, bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, rougeLsum_scores, model_sizes, model_names, rag_settings, ft_settings)

if __name__ == "__main__":
    main()