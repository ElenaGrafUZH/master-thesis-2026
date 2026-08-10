import pandas as pd
import os
from pathlib import Path
import argparse
from datetime import datetime


def extract_no_data_days():
    parser = argparse.ArgumentParser(description="Extract days with no data from merged CSV files")
    parser.add_argument("--folder", required=True, help="Path to the folder containing merged & cleaned CSV files")
    parser.add_argument("-o", "--output", help="Path to the output folder for extracted no_data_days (default: output/checks)", default="output/checks")
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
    #1. create dataframe with study id and date of no-data-days (use daily summary file to identify no-data-days)
    ##get all daily-summary-files
    daily_summary_files = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith(".csv") and "garmin-connect-daily-summary" in f.name]
    ##create dataframe
    df_daily_summary_raw = pd.DataFrame()
    for daily_summary_file in daily_summary_files:
        df = pd.read_csv(daily_summary_file)
        df_daily_summary_raw = pd.concat([df_daily_summary_raw, df], ignore_index=True)


    ##filter dataframe to only contain rows with no data (e.g. step count = 0, heart rate = 0, etc.)
    df_daily_summary_no_data = df_daily_summary_raw[(df_daily_summary_raw['averageHeartRateInBeatsPerMinute'].isna())][['study_id', 'calendarDate']]
    df_daily_summary_no_data = df_daily_summary_no_data.rename(columns={'calendarDate': 'date_no_data'})
    ##store df_daily_summary_no_data as csv file
    date = datetime.now().strftime("%Y-%m-%d")
    df_daily_summary_no_data.to_csv(output_folder / f"df_daily_summary_no_data_{date}.csv", index=False)

    print("Results successfully saved to:", output_folder / f"df_daily_summary_no_data_{date}.csv")




if __name__ == "__main__":
    extract_no_data_days()