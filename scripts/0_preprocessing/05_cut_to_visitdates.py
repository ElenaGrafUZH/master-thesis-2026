import argparse
import os
import time
import pandas as pd
from pathlib import Path
import pytz


def cut_to_visitdates():
    """
    Cuts cleaned files to only contain data in the visit date ranges
    """
    total_start_time = time.time()
    parser = argparse.ArgumentParser(description="Cut to Visit Dates")
    parser.add_argument("--folder", required=True, help="Path to the folder containing merged & cleaned CSV files")
    parser.add_argument("-o", "--output", help="Path to the output folder (default: output/cut)", default="output/cut")
    parser.add_argument("-v", "--visit_dates", help="Path to the visit dates CSV file (default: output/checks/visit_dates.csv)", default="output/checks/visit_dates.csv")
    
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
    
    # Get cleaned csv file paths
    cleaned_files = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith(".csv")]

    # Get visit dates
    df_visit_dates = pd.read_csv(args.visit_dates, encoding="utf-8")
    #create column with study_id and visit_name
    df_visit_dates['file_substring'] = df_visit_dates['study_id'].astype(str) + "_" + df_visit_dates['visit_name']

    file_substrings = df_visit_dates['file_substring'].unique()

    print(f"\n{'='*20}")
    print("\nStart Data Cutting to Visit Dates")
    len_file_substrings = len(file_substrings)

    for fs in file_substrings:
        print(f"\n{'-'*20}")
        print(f"\nProcessing files for {fs}... ({list(file_substrings).index(fs)+1}/{len_file_substrings})")
        #get visit date range for this file substring
        visit_date_range = df_visit_dates[df_visit_dates['file_substring'] == fs][['start_visit_date', 'end_visit_date']].iloc[0]
        start_visit_date = pd.Timestamp(visit_date_range['start_visit_date']).tz_localize(pytz.timezone("Europe/Zurich")).tz_convert("UTC")
        end_visit_date = (pd.Timestamp(visit_date_range['end_visit_date']) + pd.Timedelta(days=1)).tz_localize(pytz.timezone("Europe/Zurich")).tz_convert("UTC")
                                      
        #find cleaned files that contain the file substring
        files_to_cut = [f for f in cleaned_files if fs in os.path.basename(f)]
        print(f"Found {len(files_to_cut)} files to cut for {fs}.")

        for file in files_to_cut:
            print(f"\n{'*'*10}")
            print(f"Cutting file {os.path.basename(file)} to visit dates...")
            df = pd.read_csv(file, encoding="utf-8")
            #convert timestamp column to datetime
            df['datetime_utc_parsed'] = pd.to_datetime(df['datetime_utc'], utc=True, format='mixed')
            #filter rows to only include those within the visit date range
            df_cut = df[(df['datetime_utc_parsed'] >= start_visit_date) & (df['datetime_utc_parsed'] < end_visit_date)]
            df_cut = df_cut.drop(columns=['datetime_utc_parsed'])
            #save cut file to output folder with same name as original file
            output_name = os.path.basename(file).split("_merged_")[0] + "_final.csv"
            output_file = os.path.join(output_folder, os.path.basename(output_name))
            df_cut.to_csv(output_file, index=False)
            print(f"Saved cut file to {output_file}")

    print(f"\n{'='*20}")
    print("\nStart Data Cutting to Visit Dates")
    
        
   

    total_elapsed_time = time.time() - total_start_time
    print(f"\nTotal elapsed time: {total_elapsed_time:.2f} seconds")
    print("\nEnd Data Cutting to Visit Dates")
    print(f"\n{'='*20}")


if __name__ == "__main__":
    cut_to_visitdates()