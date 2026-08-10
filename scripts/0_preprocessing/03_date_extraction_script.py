

import argparse
import os
import time
import pandas as pd
from pathlib import Path


def extract_dates():
    """
    Extracts dates from merged CSV files in a specified folder and saves the
    extracted dates to an output directory. Displays progress and elapsed time.
    """
    total_start_time = time.time()
    parser = argparse.ArgumentParser(description="Extract Dates")
    parser.add_argument("--folder", required=True, help="Path to the folder containing merged & cleaned CSV files")
    parser.add_argument("-o", "--output", help="Path to the output folder for extracted dates (default: output/checks)", default="output/checks")
    args = parser.parse_args()


    script_dir = Path(__file__).resolve().parent          # .../scripts
    project_root = script_dir.parent                      # .../ (project root)

    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = (project_root / folder).resolve()
    
    # Check if output folder exists, if not create it
    output_folder = Path(args.output)
    if not output_folder.is_absolute():
        output_folder = (project_root / output_folder).resolve()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get merged csv file paths
    merged_files = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith(".csv")]

    print(f"\n{'='*20}")
    print("\nStart Date Extraction")
    df_dates = pd.DataFrame(columns=["study_id", "visit_name", "garmin_feature" ,"start_date_time", "end_date_time", "start_date", "end_date"])
    for file in merged_files:
        print(f"\n{'-'*20}")
        print(f"\nProcessing file: {file}")
        dec = os.path.basename(file).split("_")[0] + "_" + os.path.basename(file).split("_")[1]
        trial = os.path.basename(file).split("_")[2]
        garmin_feature = os.path.basename(file).split("_")[3]
        # Extract dates from the file and save to output folder
        df = pd.read_csv(file, encoding="utf-8")
        if "datetime_utc" not in df.columns:
            print(f"No datetime_utc column found in {file}. Skipping.")
            continue
        
        # Parse as UTC — safe because cleaning script always stores UTC
        dates_utc = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce").dropna()
 
        if dates_utc.empty:
            print(f"No valid datetimes found in {file}. Skipping.")
            continue
 
        # Determine local timezone for this file
        if "timezone" in df.columns:
            # Use the most common timezone in the file
            local_tz = df["timezone"].dropna().mode()
            local_tz = local_tz.iloc[0] if not local_tz.empty else "Europe/Zurich"
        else:
            local_tz = "Europe/Zurich"
 
        # Convert min/max UTC timestamps to local time
        start_local = dates_utc.min().tz_convert(local_tz).to_pydatetime()
        end_local   = dates_utc.max().tz_convert(local_tz).to_pydatetime()
 
        start_date_time = start_local.strftime("%d-%m-%Y %H:%M")
        end_date_time   = end_local.strftime("%d-%m-%Y %H:%M")
        start_date      = start_local.date()
        end_date        = end_local.date()
        df_dates = pd.concat([df_dates, pd.DataFrame([{"study_id": dec, "visit_name": trial, "garmin_feature": garmin_feature, "start_date_time": start_date_time, "end_date_time": end_date_time, "start_date": start_date, "end_date": end_date}])], ignore_index=True)
        
        
    output_file = os.path.join(args.output, "data_dates_extraction.csv")
    df_dates.to_csv(output_file, index=False)
    print("Results successfully saved to:", output_file)
    print(f"\n{'-'*20}")

    total_elapsed_time = time.time() - total_start_time
    print(f"\nTotal elapsed time: {total_elapsed_time:.2f} seconds")
    print("\nEnd Date Extraction")
    print(f"\n{'='*20}")


if __name__ == "__main__":
    extract_dates()