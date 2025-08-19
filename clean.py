import math
import pandas as pd
from datetime import datetime
import os
import calendar

input_dir = 'csv_data'
output_dir = 'cleaned_csv_data'
os.makedirs(output_dir, exist_ok=True)

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
        print(f"Header row with 'TIME' not found in {input_file}. Skipping.")
        return

    # Read the data from the correct header row
    df = pd.read_csv(input_file, skiprows=header_idx, encoding='latin1')

    # Keep only TIME and CONSUMPTION (L)
    if not set(['TIME', 'CONSUMPTION (L)']).issubset(df.columns):
        print(f"Required columns not found in {input_file}. Skipping.")
        return

    df = df[['TIME', 'CONSUMPTION (L)']]

    # Drop rows where TIME or CONSUMPTION (L) is missing or not a timestamp
    df = df.dropna(subset=['TIME', 'CONSUMPTION (L)'])
    df = df[df['TIME'].astype(str).str.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')]

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

    # If output_file exists, append new data and drop duplicates
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.drop_duplicates(subset=['Year', 'Month', 'Day', 'Date', 'Time', 'Week', 'CONSUMPTION (L)'], inplace=True)
        # Sort by Date then Time
        combined.sort_values(by=['Date', 'Time'], inplace=True)
        combined.to_csv(output_file, index=False)
        print(f"Appended and updated file: {output_file}")
    else:
        df.to_csv(output_file, index=False)
        print(f"Cleaned file saved to {output_file}")

def clean_general_csv(input_file, output_file):
    df = pd.read_csv(input_file, encoding="latin1", on_bad_lines='skip')
    # Clean column names: strip, remove quotes, handle BOM
    df.columns = [col.strip().replace('ï»¿', '').replace('"', '') for col in df.columns]
    df.rename(columns=lambda col: col[14:-4] if col.endswith(" (L)") and len(col) > 13 else col, inplace=True)
    df.fillna(0, inplace=True)

    # Find a datetime column (case-insensitive, strip spaces and quotes)
    datetime_col = None
    possible_names = ['timestamp', 'time', 'datetime', 'date']
    for col in df.columns:
        if col.strip().lower() in possible_names:
            datetime_col = col
            break
    if not datetime_col:
        print(f"No datetime column found in {input_file}. Columns found: {list(df.columns)}")
        return

    df['Timestamp'] = pd.to_datetime(df[datetime_col])
    df['Date'] = df['Timestamp'].dt.date
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month
    df['Day'] = df['Timestamp'].dt.strftime('%a')
    df['Time'] = df['Timestamp'].dt.time
    df['Month'] = df['Month'].map(lambda x: calendar.month_abbr[x])
    df['Week'] = df['Timestamp'].dt.dayofyear.apply(lambda x: math.ceil(x / 7))
    df.drop(columns=['Timestamp'], inplace=True)

    column_order = ['Year', 'Month', 'Day', 'Date', 'Time', 'Week'] + [col for col in df.columns if col not in ['Year', 'Month', 'Day', 'Date', 'Time', 'Week']]
    df = df[column_order]

    df.to_csv(output_file, index=False)
    print(f"Cleaned file saved: {output_file}")

for file_name in os.listdir(input_dir):
    if not file_name.endswith('.csv'):
        continue

    input_file_path = os.path.join(input_dir, file_name)
    # For temetra, always use tem.csv as output and append new data
    if 'temetra' in file_name.lower():
        output_file_path = os.path.join(output_dir, 'tem.csv')
        clean_temetra_logger(input_file_path, output_file_path)
    else:
        short_name = file_name[:3].lower()
        output_file_path = os.path.join(output_dir, f'{short_name}.csv')
        # For bas.csv and ind.csv, append new data and drop duplicates
        if short_name in ['bas', 'ind'] and os.path.exists(output_file_path):
            # Clean new data
            df_new = pd.read_csv(input_file_path, encoding="latin1", on_bad_lines='skip')
            df_new.columns = [col.strip().replace('ï»¿', '').replace('"', '') for col in df_new.columns]
            df_new.rename(columns=lambda col: col[14:-4] if col.endswith(" (L)") and len(col) > 13 else col, inplace=True)
            df_new.fillna(0, inplace=True)

            datetime_col = None
            possible_names = ['timestamp', 'time', 'datetime', 'date']
            for col in df_new.columns:
                if col.strip().lower() in possible_names:
                    datetime_col = col
                    break
            if not datetime_col:
                print(f"No datetime column found in {input_file_path}. Columns found: {list(df_new.columns)}")
                continue

            df_new['Timestamp'] = pd.to_datetime(df_new[datetime_col])
            df_new['Date'] = df_new['Timestamp'].dt.date
            df_new['Year'] = df_new['Timestamp'].dt.year
            df_new['Month'] = df_new['Timestamp'].dt.month
            df_new['Day'] = df_new['Timestamp'].dt.strftime('%a')
            df_new['Time'] = df_new['Timestamp'].dt.time
            df_new['Month'] = df_new['Month'].map(lambda x: calendar.month_abbr[x])
            df_new['Week'] = df_new['Timestamp'].dt.dayofyear.apply(lambda x: math.ceil(x / 7))
            df_new.drop(columns=['Timestamp'], inplace=True)

            column_order = ['Year', 'Month', 'Day', 'Date', 'Time', 'Week'] + [col for col in df_new.columns if col not in ['Year', 'Month', 'Day', 'Date', 'Time', 'Week']]
            df_new = df_new[column_order]

            # Read existing cleaned file
            df_existing = pd.read_csv(output_file_path)
            # Combine and drop duplicates
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.drop_duplicates(inplace=True)
            df_combined.to_csv(output_file_path, index=False)
            print(f"Appended and updated file: {output_file_path}")
        else:
            clean_general_csv(input_file_path, output_file_path)

