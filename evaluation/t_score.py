import json
import os
from scipy import stats
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import math

# Directory containing the JSON files
dir_path = '../eval_results'

# Initialize dictionaries to hold metric values categorized by model and conditions
metrics = {}

# Read data from JSON files and categorize based on model, rag, and ft status
for file_name in os.listdir(dir_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)
            model_name = data['model_name']
            use_rag = data['use_rag']
            use_ft = data['use_ft']
            model_size = data['model_size']

            key = (use_ft, use_rag)

            if model_name not in metrics:

                metrics[model_name] = {
                    'model_name': model_name,
                    'model_size': float(model_size.replace('B', '')),
                    (False, False): {'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []},
                    (False, True): {'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []},
                    (True, False): {'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []},
                    (True, True): {'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []},
                }

            for result in data['results']:
                metrics[model_name][key]['bleu'].append(result['bleu_score'])
                metrics[model_name][key]['rouge1'].append(
                    result['rouge_scores']['rouge1'])
                metrics[model_name][key]['rouge2'].append(
                    result['rouge_scores']['rouge2'])
                metrics[model_name][key]['rougeL'].append(
                    result['rouge_scores']['rougeL'])
                metrics[model_name][key]['rougeLsum'].append(
                    result['rouge_scores']['rougeLsum'])

# Function to perform t-tests


def perform_t_test(base_group, comparison_group):
    t_stat, p_value = stats.ttest_ind(
        base_group, comparison_group, equal_var=False)
    
    df = len(base_group) + len(comparison_group) - 2
    r = math.sqrt((t_stat**2) / (t_stat**2 + df))

    if r < 0.3:
        significance = "weak"
    elif r < 0.5:
        significance = "moderate"
    else:
        significance = "strong"
    

    return (t_stat, p_value, significance, r)


# Perform t-tests for each model
base_condition = (False, False)
comparison_conditions = [
    (True, False),
    (False, True),
    (True, True)
]

rows = []

for model_name, data in metrics.items():
    for comp_cond in comparison_conditions:
        ft, rag = comp_cond

        t_stat, p_val, significance_bleu, r = perform_t_test(
            data[base_condition]['bleu'], data[comp_cond]['bleu'])

        # drop rows with NaN values
        if np.isnan(t_stat):
            continue

        bleu = f'{t_stat:.3f} ({p_val:.3f}) [{r:.3f}]'

        t_stat, p_val, significance_rouge1, r = perform_t_test(
            data[base_condition]['rouge1'], data[comp_cond]['rouge1'])
        rouge1 = f'{t_stat:.3f} ({p_val:.3f}) [{r:.3f}]'

        t_stat, p_val, significance_rouge2, r = perform_t_test(
            data[base_condition]['rouge2'], data[comp_cond]['rouge2'])
        rouge2 = f'{t_stat:.3f} ({p_val:.3f}) [{r:.3f}]'

        t_stat, p_val, significance_rougeL, r = perform_t_test(
            data[base_condition]['rougeL'], data[comp_cond]['rougeL'])
        rougeL = f'{t_stat:.3f} ({p_val:.3f}) [{r:.3f}]'

        t_stat, p_val, significance_rougeLsum, r = perform_t_test(
            data[base_condition]['rougeLsum'], data[comp_cond]['rougeLsum'])
        rougeLsum = f'{t_stat:.3f} ({p_val:.3f}) [{r:.3f}]'

        rows.append([data['model_size'], model_name, ft, rag, bleu, rouge1, rouge2, rougeL, rougeLsum,
                    significance_bleu, significance_rouge1, significance_rouge2, significance_rougeL, significance_rougeLsum])


rows = sorted(rows, key=lambda x: (-x[0], x[1][::-1], x[2], x[3]))

headers = ['Model Size', 'Model Name', 'FT', 'RAG', 'BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum', 
           'BLEU Significance', 'ROUGE-1 Significance', 'ROUGE-2 Significance', 'ROUGE-L Significance', 'ROUGE-Lsum Significance']
column_widths = [-1, 200, 30, 30, 125, 125, 125, 125, 125, -1, -1, -1, -1, -1]


fill_colors = [['white'] * len(headers) for _ in range(len(rows))]
colorToggle = True
lastModel = ""
for row_idx, row in enumerate(rows):
    # Alternate Color
    if row[1] != lastModel:
        colorToggle = not colorToggle
    for col in range(0, 4):
        fill_colors[row_idx][col] = '#e1e1e1' if colorToggle else 'white'

    lastModel = row[1]

    # Highlight significant values
    for col in range(4, 9):
        significance_col = col + 5
        if row[significance_col] == 'moderate':
            fill_colors[row_idx][col] = 'lightyellow'
        elif row[significance_col] == 'strong':
            fill_colors[row_idx][col] = 'lightgreen'
        else:
            fill_colors[row_idx][col] = 'white'


for row_idx, row in enumerate(rows):
    for col in range(2, 4):
        row[col] = '✓' if row[col] else 'X'

fill_colors = np.array(fill_colors).T.tolist()
rows = np.array(rows).T.tolist()

# Plotly table
fig = make_subplots(rows=1, cols=1, specs=[[{"type": "table"}]])
fig.add_trace(go.Table(
    columnwidth=column_widths,
    header=dict(values=headers,
                fill_color='#e1e1e1',
                align='center'),
    cells=dict(values=rows,
               fill_color=fill_colors,
               align='center',
               )
))

# Show the table
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.show()
fig.write_image("t_score.png", format='png', engine='kaleido',
                scale=5, width=1300, height=607)
