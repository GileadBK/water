import math
import pandas as pd
from datetime import datetime
import os
import calendar
import matplotlib

input_dir='csv_data' 
output_dir='cleaned_csv_data'

os.makedirs(output_dir, exist_ok=True)

for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):  
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, f'{file_name[:3]}.csv')
        
        if os.path.exists(output_file_path):
            print(f"File already cleaned, skipping: {file_name}")
            continue

        df = pd.read_csv(input_file_path, encoding="latin1", on_bad_lines='skip')
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
            print(f"No datetime column found in {file_name}. Columns found: {list(df.columns)}")
            continue

        df['Timestamp'] = pd.to_datetime(df[datetime_col])
        df['Date'] = df['Timestamp'].dt.date
        df['Year'] = df['Timestamp'].dt.year
        df['Month'] = df['Timestamp'].dt.month
        df['Day'] = df['Timestamp'].dt.strftime('%a')
        df['Time'] = df['Timestamp'].dt.time 

        df['Month'] = df['Month'].map(lambda x: calendar.month_abbr[x])

        df['Week'] = df['Timestamp'].dt.dayofyear.apply(lambda x: math.ceil(x / 7))

        df.drop(columns=['Timestamp'], inplace=True)

        column_order = ['Year', 'Month', 'Day', 'Date', 'Time','Week'] + [col for col in df.columns if col not in ['Year', 'Month', 'Day', 'Date', 'Time', 'Week']]
        df = df[column_order]

        if os.path.exists(output_file_path):
            existing_df = pd.read_csv(output_file_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates()
            combined_df.to_csv(output_file_path, index=False)
            print(f"Appended non-duplicate cleaned data to: {output_file_path}")
        else:
            df.to_csv(output_file_path, index=False)
            print(f"Cleaned file saved: {output_file_path}")

