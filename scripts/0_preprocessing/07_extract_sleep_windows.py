import argparse
import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime


def extract_sleep_windows():
    """
    Extracts sleep windows from merged CSV files in a specified folder and saves the
    extracted windows to an output directory. Displays progress and elapsed time.
    """
    total_start_time = time.time()
    parser = argparse.ArgumentParser(description="Extract Sleep Windows")
    parser.add_argument("--folder", required=True, help="Path to the folder containing merged & cleaned CSV files")
    parser.add_argument("-o", "--output", help="Path to the output folder for extracted sleep_windows (default: output/checks)", default="output/checks")
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
    merged_files = [f.path for f in os.scandir(args.folder) if f.is_file() and f.name.endswith(".csv") and "garmin-connect-sleep-summary" in f.name]

    print(f"\n{'='*20}")
    print(f"\nStart Sleep Window Extractions for {len(merged_files)} files")
    df_sleep_windows = pd.DataFrame(columns=["study_id", "onset_time", "wake_up_time", 'durationInMs', 'timezone'])
    counter = 1
    for file in merged_files:
        print(f"\n{'-'*20}")
        print(f"\nProcessing file: {file} ({counter}/{len(merged_files)})")
        dec = os.path.basename(file).split("_")[0] + "_" + os.path.basename(file).split("_")[1]
        # Extract sleep windows from the file and save to output folder
        df = pd.read_csv(file, encoding="utf-8")
        if df.empty:
            print(f"No data found in {file}. Skipping.")
            continue
        dt_utc = pd.to_datetime(
            df["datetime_utc"],
            errors="coerce",
            utc=True
        )

        df['study_id'] = dec
        df['onset_time'] = [
            ts.tz_convert(tz) if pd.notna(ts) and pd.notna(tz) else pd.NaT
            for ts, tz in zip(dt_utc, df["timezone"])
        ]
        df['wake_up_time'] = df['onset_time']+ pd.to_timedelta(df['durationInMs'], unit='ms')
        df_sleep_windows = pd.concat([df_sleep_windows, df[['study_id', 'onset_time', 'wake_up_time', 'durationInMs', 'timezone']]], ignore_index=True)
        counter += 1

    date = datetime.now().strftime("%Y-%m-%d")
    output_file = os.path.join(args.output, f"data_sleep_windows_extraction_{date}.csv")
    df_sleep_windows.to_csv(output_file, index=False)
    print("Results successfully saved to:", output_file)
    print(f"\n{'-'*20}")

    total_elapsed_time = time.time() - total_start_time
    print(f"\nTotal elapsed time: {total_elapsed_time:.2f} seconds")
    print("\nEnd Sleep Window Extraction")
    print(f"\n{'='*20}")


if __name__ == "__main__":
    extract_sleep_windows()