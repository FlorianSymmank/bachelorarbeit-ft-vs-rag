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
        return float(model_size_str.replace('B', '')) * 10
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

    original_data_len = len(durations)

    # Filter out the 1% min and max outliers
    def filter_outliers(data):
        lower_bound = np.percentile(data, .15)
        upper_bound = np.percentile(data, 99.85)
        return [x for x in data if lower_bound <= x <= upper_bound]

    filtered_durations = filter_outliers(durations)
    filtered_bleu_scores = [bleu_scores[i] for i in range(len(durations)) if durations[i] in filtered_durations]
    filtered_rouge1_scores = [rouge1_scores[i] for i in range(len(durations)) if durations[i] in filtered_durations]
    filtered_rouge2_scores = [rouge2_scores[i] for i in range(len(durations)) if durations[i] in filtered_durations]
    filtered_rougeL_scores = [rougeL_scores[i] for i in range(len(durations)) if durations[i] in filtered_durations]
    filtered_rougeLsum_scores = [rougeLsum_scores[i] for i in range(len(durations)) if durations[i] in filtered_durations]
    filtered_model_sizes = [model_sizes[i] for i in range(len(durations)) if durations[i] in filtered_durations]
    filtered_model_names = [model_names[i] for i in range(len(durations)) if durations[i] in filtered_durations]
    filtered_rag_settings = [rag_settings[i] for i in range(len(durations)) if durations[i] in filtered_durations]
    filtered_ft_settings = [ft_settings[i] for i in range(len(durations)) if durations[i] in filtered_durations]

    filtered_data_len = len(filtered_durations)
    removed_data_points = original_data_len - filtered_data_len

    return (filtered_durations, filtered_bleu_scores, filtered_rouge1_scores, filtered_rouge2_scores,
            filtered_rougeL_scores, filtered_rougeLsum_scores, filtered_model_sizes, filtered_model_names,
            filtered_rag_settings, filtered_ft_settings, removed_data_points)

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

def plot_data(durations, scores, score_type, model_sizes, model_names, rag_settings, ft_settings, removed_data_points):
    """Plot the extracted data."""
    color_map, unique_model_names = get_color_map(model_names)
    colors = [color_map[model_name] for model_name in model_names]

    plt.figure(figsize=(10, 6))

    for rag, ft in [(True, True), (True, False), (False, True), (False, False)]:
        indices = [i for i in range(len(rag_settings)) if rag_settings[i] == rag and ft_settings[i] == ft]
        sub_durations = [durations[i] for i in indices]
        sub_scores = [scores[i] for i in indices]
        sub_model_sizes = [model_sizes[i] for i in indices]
        sub_colors = [colors[i] for i in indices]
        marker = get_marker(rag, ft)

        plt.scatter(sub_durations, sub_scores, s=sub_model_sizes, c=sub_colors, marker=marker, label=f'RAG={rag}, FT={ft}')

    plt.title(f'Question Time vs {score_type} Score (Point Size by Model Size)')
    plt.xlabel('Duration (ms) [log scale]')
    plt.ylabel(f'{score_type} Score')

    # Create custom legend for colors
    sorted_model_names = sorted(unique_model_names)
    color_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[model_name], markersize=10) for model_name in sorted_model_names]
    # Create custom legend for markers
    marker_labels = ['RAG & FT', 'RAG', 'FT', 'Base']
    marker_styles = ['o', 's', '^', 'D']
    marker_handles = [plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray', markersize=10) for marker in marker_styles]

    # Combine both legends
    handles = color_handles + marker_handles
    labels = sorted_model_names + marker_labels

    plt.legend(handles, labels, loc='upper right')
    plt.ylim(0, 1)
    plt.xscale('log')

    # Compute the range of durations to avoid empty area
    min_duration = min(durations)
    max_duration = max(durations)
    plt.xlim(min_duration * 0.9, max_duration * 1.1)

    # Fit a polynomial trendline
    poly_coeffs = np.polyfit(np.log10(durations), scores, 1)
    poly_eq = np.poly1d(poly_coeffs)
    trendline_x = np.linspace(min_duration, max_duration, 100)
    trendline_y = poly_eq(np.log10(trendline_x))

    plt.plot(trendline_x, trendline_y, color='black', linestyle='--', linewidth=1, label='Trendline')
    plt.legend(handles + [plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1)], labels + ['Trendline'], loc='upper right')

    # Add text box with information about removed data points
    textstr = f'Removed {removed_data_points} outliers (0.15% min and max)'
    plt.gcf().text(0.072, 0.9275, textstr, fontsize=10, verticalalignment='top', bbox=dict( facecolor="wheat", edgecolor="black"))

    plt.tight_layout()
    plt.show()

def main():
    data_list = read_json_files(eval_results_dir)
    (durations, bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, rougeLsum_scores, model_sizes, model_names, 
     rag_settings, ft_settings, removed_data_points) = extract_data(data_list)
    
    # Define the score type to plot here
    score_type = 'ROUGE-1'  # Change this to 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', or 'ROUGE-Lsum' as needed
    score_map = {
        'BLEU': bleu_scores,
        'ROUGE-1': rouge1_scores,
        'ROUGE-2': rouge2_scores,
        'ROUGE-L': rougeL_scores,
        'ROUGE-Lsum': rougeLsum_scores
    }

    if score_type in score_map:
        scores = score_map[score_type]
        plot_data(durations, scores, score_type, model_sizes, model_names, rag_settings, ft_settings, removed_data_points)
    else:
        print(f"Invalid score type: {score_type}. Please choose from 'BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', or 'ROUGE-Lsum'.")

if __name__ == "__main__":
    main()