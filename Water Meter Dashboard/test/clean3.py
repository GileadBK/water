import math
import pandas as pd
from datetime import datetime
import os
import calendar

def clean_temetra_logger(input_file, output_file):
    # Find the header row (the one starting with 'TIME')
    with open(input_file, 'r', encoding='latin1') as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('TIME,'):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Header row with 'TIME' not found.")

    # Read the data from the correct header row
    df = pd.read_csv(input_file, skiprows=header_idx, encoding='latin1')

    # Keep only TIME and CONSUMPTION (L)
    df = df[['TIME', 'CONSUMPTION (L)']]

    # Drop rows where TIME or CONSUMPTION (L) is missing or not a timestamp
    df = df.dropna(subset=['TIME', 'CONSUMPTION (L)'])
    df = df[df['TIME'].str.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')]

    # Parse TIME into components
    df['TIME'] = pd.to_datetime(df['TIME'])
    df['Year'] = df['TIME'].dt.year
    df['Month'] = df['TIME'].dt.month
    df['Day'] = df['TIME'].dt.strftime('%a')
    df['Date'] = df['TIME'].dt.date
    df['Time'] = df['TIME'].dt.time
    df['Week'] = df['TIME'].dt.dayofyear.apply(lambda x: math.ceil(x / 7))
    df['Month'] = df['Month'].map(lambda x: calendar.month_abbr[x])

    # Reorder columns
    df = df[['Year', 'Month', 'Day', 'Date', 'Time', 'Week', 'CONSUMPTION (L)']]

    # Save to cleaned_csv_data
    df.to_csv(output_file, index=False)
    print(f"Cleaned file saved to {output_file}")
    return df

# Input and output paths
input_dir = 'csv_data'
output_dir = 'cleaned_csv_data'
for file_name in os.listdir(input_dir):
    if file_name.__contains__('temetra'):
        input_file = os.path.join(f'csv_data', file_name)
        output_folder = 'cleaned_csv_data'
        output_file = os.path.join(output_folder, f'{file_name}_cleaned.csv')

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Clean the data
clean_temetra_logger(input_file, output_file)