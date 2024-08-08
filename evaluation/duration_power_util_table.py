import os
import json
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import locale

# Set locale to use commas as decimal separators
locale.setlocale(locale.LC_NUMERIC, 'de_DE.UTF-8')  # German locale

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

# Function to convert milliseconds to minutes:seconds format
def ms_to_min_sec(ms):
    minutes = int(ms // 60000)
    seconds = int((ms % 60000) // 1000)
    return f"{minutes}:{seconds:02d}"

# Function to convert milliseconds to hours
def ms_to_hours(ms):
    return ms / 3600000

# Function to create a DataFrame
def create_dataframe(json_data):
    rows = []
    for entry in json_data:
        duration_sum = sum(result.get('duration_ms', 0) for result in entry.get('results', []))
        avg_power = entry.get("mean_avg_power")
        total_power = avg_power * ms_to_hours(duration_sum)  # Calculate total consumed power
        row = {
            "Model Size": entry.get("model_size"),  # Add model_size column
            "Modelname": entry.get("model_name"),
            "FT": "✓" if entry.get("use_ft") else "X",
            "RAG": "✓" if entry.get("use_rag") else "X",
            "Total Duration (min:sec)": ms_to_min_sec(duration_sum),
            "Avg Power (W)": avg_power,
            "Avg Utilization (%)": entry.get("mean_avg_utilization"),
            "Total Consumed Power (Wh)": total_power  # New column
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Model Size", "Modelname", "FT", "RAG"], ascending=[False, False, True, True])
    return df

# Load JSON data
json_data = load_data(directory)

# Create DataFrame
df = create_dataframe(json_data)

# Format to use commas as decimal separators and show only 3 decimal places
df = df.map(lambda x: locale.format_string("%.2f", x) if isinstance(x, float) else x)

# Columns to highlight
columns_to_highlight = ["Total Duration (min:sec)", "Avg Power (W)", "Avg Utilization (%)", "Total Consumed Power (Wh)"]

# Create a list of fill colors for each cell
fill_colors = [['white'] * len(df) for _ in df.columns]

# Highlight the minimum and maximum values
for col in columns_to_highlight:
    if col == "Total Duration (min:sec)":
        # Convert time to numeric for comparison
        durations = []
        for val in df[col]:
            minute, sec = map(int, val.split(':'))
            durations.append(minute * 60 + sec)
        min_val = min(durations)
        max_val = max(durations)
        min_indices = [i for i, x in enumerate(durations) if x == min_val]
        max_indices = [i for i, x in enumerate(durations) if x == max_val]
    else:
        vals = []
        for val in df[col]:
            vals.append(locale.atof(val))

        min_val = min(vals)
        max_val = max(vals)
        min_indices = [i for i, x in enumerate(vals) if x == min_val]
        max_indices = [i for i, x in enumerate(vals) if x == max_val]

    col_idx = df.columns.get_loc(col)
    for row_idx in min_indices:
        fill_colors[col_idx][row_idx] = 'lightgreen'
    for row_idx in max_indices:
        fill_colors[col_idx][row_idx] = 'lightcoral'

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
average_duration_sec = sum([int(val.split(':')[0]) * 60 + int(val.split(':')[1]) for val in df["Total Duration (min:sec)"]]) / len(df)
average_duration = ms_to_min_sec(average_duration_sec * 1000)
average_power = df["Avg Power (W)"].apply(locale.atof).mean()
average_utilization = df["Avg Utilization (%)"].apply(locale.atof).mean()
total_consumed_power = df["Total Consumed Power (Wh)"].apply(locale.atof).mean()

# Append averages to the DataFrame
average_row = {
    "Model Size": "",
    "Modelname": "Average",
    "FT": "",
    "RAG": "",
    "Total Duration (min:sec)": average_duration,
    "Avg Power (W)": locale.format_string("%.2f", average_power),
    "Avg Utilization (%)": locale.format_string("%.2f", average_utilization),
    "Total Consumed Power (Wh)": locale.format_string("%.2f", total_consumed_power)
}

df = pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)
for col in df.columns:
    col_idx = df.columns.get_loc(col)
    fill_colors[col_idx].append('#c7c7c7')

column_widths = [-1, 400, 50, 50, 150, 150, 150, 150]

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
fig.write_image("model_metrics_overview.png", format='png', engine='kaleido', scale=5, width=1000, height=910)