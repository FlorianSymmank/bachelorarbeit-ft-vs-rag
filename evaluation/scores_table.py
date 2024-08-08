import os
import json
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Directory containing JSON files
directory = '../eval_results/'

# Function to load data from JSON files
def load_data(directory):
    json_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                json_data.append(data)
    return json_data

# Function to create a DataFrame
def create_dataframe(json_data):
    rows = []
    for entry in json_data:
        row = {
            "Model Size": entry.get("model_size"),  # Add model_size column
            "Modelname": entry.get("model_name"),
            "FT": "✓" if entry.get("use_ft") else "X",
            "RAG": "✓" if entry.get("use_rag") else "X",
            "BLEU Score": entry.get("mean_bleu_score"),
            "ROUGE-1": entry["mean_rouge_scores"].get("rouge1"),
            "ROUGE-2": entry["mean_rouge_scores"].get("rouge2"),
            "ROUGE-L": entry["mean_rouge_scores"].get("rougeL"),
            "ROUGE-Lsum": entry["mean_rouge_scores"].get("rougeLsum"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Model Size", "Modelname", "FT", "RAG"], ascending=[False, False, True, True])
    return df

# Load JSON data
json_data = load_data(directory)

# Create DataFrame
df = create_dataframe(json_data)

# Format to show only 3 decimal places
df = df.round(3)

# Columns to highlight
columns_to_highlight = ["BLEU Score", "ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum"]

# Create a list of fill colors for each cell
fill_colors = [['white'] * len(df) for _ in df.columns]

# Highlight the maximum values with grey background
for col in columns_to_highlight:
    max_val = df[col].max()
    max_indices = df.index[df[col] == max_val].tolist()
    col_idx = df.columns.get_loc(col)
    for row_idx in max_indices:
        fill_colors[col_idx][row_idx] = 'lightgreen'

# Apply alternating row colors based on "Model Name"
model_names = df['Modelname'].tolist()
unique_model_names = list(dict.fromkeys(model_names))  # Preserve order and remove duplicates

row_colors = []
color_toggle = True
for model_name in model_names:
    if model_name == unique_model_names[0]:
        row_colors.append('white' if color_toggle else '#e1e1e1')
    else:
        color_toggle = not color_toggle  # Toggle color on new model name
        unique_model_names.pop(0)
        row_colors.append('white' if color_toggle else '#e1e1e1')

# Merge row colors and highlighted cells
for col_idx in range(len(fill_colors)):
    for row_idx in range(len(fill_colors[col_idx])):
        if fill_colors[col_idx][row_idx] == 'white':
            fill_colors[col_idx][row_idx] = row_colors[row_idx]

# Calculate averages
average_bleu = df["BLEU Score"].mean()
average_rouge1 = df["ROUGE-1"].mean()
average_rouge2 = df["ROUGE-2"].mean()
average_rougeL = df["ROUGE-L"].mean()
average_rougeLsum = df["ROUGE-Lsum"].mean()

# Append averages to the DataFrame
average_row = {
    "Model Size": "",
    "Modelname": "Average",
    "FT": "",
    "RAG": "",
    "BLEU Score": round(average_bleu, 3),
    "ROUGE-1": round(average_rouge1, 3),
    "ROUGE-2": round(average_rouge2, 3),
    "ROUGE-L": round(average_rougeL, 3),
    "ROUGE-Lsum": round(average_rougeLsum, 3)
}

df = pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)
for col in df.columns:
    col_idx = df.columns.get_loc(col)
    fill_colors[col_idx].append('#c7c7c7')

column_widths = [-1, 400, 50, 50, 150, 150, 150, 150, 150] 

# Plotly table
fig = make_subplots(rows=1, cols=1, specs=[[{"type": "table"}]])
fig.add_trace(go.Table(
    columnwidth=column_widths,
    header=dict(values=list(df.columns),
                fill_color='#e1e1e1',
                align='center'),    
    cells=dict(values=[df[col].tolist() for col in df.columns],
               fill_color=fill_colors,
               align='center'))   
)

# Show the table
# fig.show()
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image("scores_overview.png", format='png', engine='kaleido', scale=5, width=1000, height=910)