import os
import json
import matplotlib.pyplot as plt
import math


def load_data(directory, model_name):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json") and model_name in filename:
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                json_data = json.load(file)
                data.append(json_data)
    return data


def plot_gpu_metrics(data, question_index, main_title):
    num_plots = len(data)
    if num_plots == 0:
        print("No data to plot.")
        return

    # Calculate the number of rows and columns
    num_cols = 2 if num_plots > 1 else 1
    num_rows = math.ceil(num_plots / num_cols)

    # Determine the maximum duration for the x-axis
    max_duration = 0
    for d in data:
        gpu_metrics = d['results'][question_index]['gpu_metrics_100ms']
        max_duration = max(max_duration, len(gpu_metrics))

    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(14, 5 * num_rows))
    axes = axes.flatten() if num_plots > 1 else [axes]

    fig.suptitle(main_title, fontsize=16)

    for idx, d in enumerate(data):
        model_name = d["model_name"]
        use_ft = d["use_ft"]
        use_rag = d["use_rag"]
        config_label = f'FT:{use_ft} RAG:{use_rag}'

        # Extract GPU metrics data from the specified question
        gpu_metrics = d['results'][question_index]['gpu_metrics_100ms']

        # Separate power and utilization values
        power_values = [metric[0] for metric in gpu_metrics]
        utilization_values = [metric[1] for metric in gpu_metrics]

        # Create a time series
        time_series = range(len(gpu_metrics))

        ax1 = axes[idx]
        ax1.set_xlabel('Time (100ms intervals)')
        ax1.set_ylabel('Power (W)', color='tab:blue')
        ax1.plot(time_series, power_values,
                 color='tab:blue', label='Power (W)')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0, 230)
        ax1.set_xlim(0, max_duration)

        # Create a second y-axis to plot GPU utilization
        ax2 = ax1.twinx()
        ax2.set_ylabel('GPU Utilization (%)', color='tab:orange')
        ax2.plot(time_series, utilization_values, color='tab:orange',
                 linestyle='--', label='GPU Utilization (%)')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.set_ylim(0, 100)

        # Add title and grid
        ax1.set_title(config_label)
        ax1.grid(True)

    # Hide any unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


models = [
    "Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Qwen2-7B-Instruct",
    "Qwen-7B-Chat",
    "Mistral-7B-Instruct-v0.3",
    "Phi-3-mini-128k-instruct",
    "Qwen1.5-4B-Chat",
    "internlm2-chat-1_8b",
    "Qwen2-1.5B-Instruct",
    "Qwen1.5-0.5B-Chat",
    "Qwen2-0.5B-Instruct",
    "Qwen1.5-1.8B-Chat",
    "stablelm-2-1_6b-chat",
    "SmolLM-1.7B-Instruct",
]

directory = "../eval_results"
for model_name in models:
    data = load_data(directory, model_name)

    # Plot GPU metrics for each model variant
    for question_index in range(20):
        main_title = f'GPU Metrics for {model_name} - Question {question_index + 1}'
        plot_gpu_metrics(data, question_index, main_title)