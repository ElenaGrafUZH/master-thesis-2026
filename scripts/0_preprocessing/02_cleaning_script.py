import os
import argparse
import time
import pandas as pd
from pathlib import Path
import pytz


def data_cleaning():
    """
    Cleans merged CSV files in a specified folder by converting timestamps,
    handling missing values, and saving the cleaned data to an output
    directory.
    """
    total_start_time = time.time()
    parser = argparse.ArgumentParser(description="Clean Data")
    parser.add_argument("--folder", required=True, help="Folder containing merged .csv files (path to output folder)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent          # .../scripts
    project_root = script_dir.parent                      # .../ (project root)

    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = (project_root / folder).resolve()

    #check if output/cleaned folder exists, if not create it    
    output_cleaned_folder = os.path.join(folder, "cleaned")
    if not os.path.exists(output_cleaned_folder):
        os.makedirs(output_cleaned_folder)

    merged_files = [str(p) for p in folder.iterdir() if p.is_file() and p.suffix == ".csv"]

    DEFAULT_TZ = "Europe/Zurich"
    valid_tzs = set(pytz.all_timezones)

    #get merged csv file paths
    # merged_files = [f.path for f in os.scandir(args.folder) if f.is_file() and f.name.endswith(".csv")]
    print(f"\n{'='*20}")
    print("\nStart Data cleaning")
    for file in merged_files:
        print(f"\n{'-'*20}")
        print(f"\nProcessing file: {file}")
        df = pd.read_csv(file, encoding="utf-8")
        #check if unixTimestampInMs column exists

        if "unixTimestampInMs" not in df.columns:
            if "calendarDate" not in df.columns:
                print(f"Neither unixTimestampInMs nor calendarDate found in {file}. Skipping.")
                continue
            # calendarDate is local date only (no time) — localize to default TZ so it's consistent
            df['datetime_utc'] = pd.to_datetime(df['calendarDate']).dt.tz_localize(DEFAULT_TZ, ambiguous='infer', nonexistent='shift_forward').dt.tz_convert("UTC")
            # Add timezone column so downstream scripts know the local TZ
            if 'timezone' not in df.columns:
                df['timezone'] = DEFAULT_TZ
            # add datetime column in local time for easier cutting to visit dates later
            df['datetime'] = df['datetime_utc'].dt.tz_convert(DEFAULT_TZ).dt.strftime('%Y-%m-%dT%H:%M:%S%z')
        else:
            # Convert unix ms → UTC datetime
            df['datetime_utc'] = pd.to_datetime(df['unixTimestampInMs'], unit='ms', utc=True)
 
            if 'timezone' in df.columns:
                # Validate and fill missing timezones
                df['timezone'] = df['timezone'].fillna(DEFAULT_TZ)
                invalid = df[~df['timezone'].isin(valid_tzs)]['timezone'].unique()
                if len(invalid):
                    print(f"Warning: invalid timezones found, falling back to {DEFAULT_TZ}: {invalid}")
                    df['timezone'] = df['timezone'].where(df['timezone'].isin(valid_tzs), DEFAULT_TZ)
            else:
                df['timezone'] = DEFAULT_TZ

            converted = []
            for tz, group in df.groupby('timezone'):
                group = group.copy()
                group['datetime'] = group['datetime_utc'].dt.tz_convert(tz).dt.strftime('%Y-%m-%dT%H:%M:%S%z')
                converted.append(group)
            df = pd.concat(converted).sort_index()

 
        #handle empty strings
        ##check if empty strings exist in the dataframe
        if (df == "").any().any():
            print(f"Empty strings found in {file}. Replacing with NaN.")
            df.replace("", pd.NA, inplace=True)

        #save cleaned data
        output = file.replace(".csv", "_cleaned.csv")
        output = os.path.join(output_cleaned_folder, os.path.basename(output))
        df.to_csv(output, index=False, encoding="utf-8")

        
    total_elapsed_time = time.time() - total_start_time
    print(f"\nTotal elapsed time: {total_elapsed_time:.2f} seconds")
    print("\nEnd Data cleaning")
    print(f"\n{'='*20}")


if __name__ == "__main__":
    data_cleaning()