import argparse
import os
import json
import time
from pathlib import Path
import configparser
import requests


import numpy as np
import pandas as pd


config = configparser.ConfigParser()
config.read("settings.ini")



def api_file_upload(file_path, field_name, study_id, event_name):
    """
    Uploads a file to REDCap using the API.
    Args:
        file (str): Path to the file to be uploaded.
        field_name (str): The REDCap field name where the file should be uploaded.
        study_id (str): The study ID associated with the file.
        event_name (str): The REDCap event name associated with the file.
    """
    
    print(f"Uploading {file_path.split('_')[3]} to REDCap with field name {field_name}")
    
    data = {
        'token': config['REDCap']['api_token'],
        'content': 'file',
        'action': 'import',
        'format': 'json',
        'returnFormat': 'json',
        'record': study_id,
        'field': field_name,
        'event': event_name
    }
    # Open the PDF file in binary mode and prepare for upload
    files = {
        'file': (file_path, open(file_path, 'rb'))
    }
    # print(f"Data to be sent: {data}, {files}")

    #POST request to upload file
    r = requests.post(config['REDCap']['api_url'], data=data, files=files, timeout=1000)

  
    if r.status_code == 200:
        print(f"Success: File {file_path} uploaded successfully.")
        print(f"\n{'-'*20}")
    else:
        print('Error uploading the file:', r.status_code)
        print('Response:', r.text)
        print(f"\n{'-'*20}")
    
    return r.status_code

def api_date_upload(study_id, event_name, start_date, end_date):
    """
    Uploads start and end dates to REDCap using the API.
    Args:
        study_id (str): The study ID associated with the dates.
        event_name (str): The REDCap event name associated with the dates.
        start_date (str): The start date to be uploaded.
        end_date (str): The end date to be uploaded.
    """

    data = {
        'token': config['REDCap']['api_token'],
        'content': 'record',
        'format': 'json',
        'type': 'flat',
        'overwriteBehavior': 'overwrite',
        'data': json.dumps([{
            'study_id': study_id,
            'redcap_event_name': event_name,
            'venu3_start': start_date,
            'venu3_end': end_date
        }])
    }

    r = requests.post(config['REDCap']['api_url'], data=data, timeout=120)
    if r.status_code == 200:
        print("Success: Dates uploaded successfully.")
        print(f"\n{'-'*20}")
    else:
        print('Error uploading the dates:', r.status_code)
        print('Response:', r.text)
        print(f"\n{'-'*20}")
    
    return r.status_code


def upload_to_redcap():
    """
    Upload cleaned CSV files to REDCap using the API. Displays progress and elapsed time.
    """
    #for each study ID get all csv files per visit (V1 and V2) plus smartwatch start and end date time
    #for V1 data:
    #Event: Baseline_Period (Arm 1: Main), Record ID: study ID
    #for V2 data:
    #Event: Intervention_Period (Arm 1: Main), Record ID: study ID

    total_start_time = time.time()
    parser = argparse.ArgumentParser(description="Upload CSV to REDCap")
    parser.add_argument("--folder", required=True, help="Path to the folder containing merged & cleaned CSV files")
    parser.add_argument("--redcap_fields", required=True, help="Path to the REDCap fields mapping file")
    parser.add_argument("--dates", required=True, help="Path to the dates file of 3_date_extraction step")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent          # .../scripts
    project_root = script_dir.parent                      # .../ (project root)

    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = (project_root / folder).resolve()

    #get field names - file names mapping
    redcap_fields_file = Path(args.redcap_fields)
    if not redcap_fields_file.is_absolute():
        redcap_fields_file = (project_root / redcap_fields_file).resolve()
    filed_mapping = pd.read_csv(redcap_fields_file, encoding="utf-8")

    #get dates file path
    dates_file = Path(args.dates)
    if not dates_file.is_absolute():
        dates_file = (project_root / dates_file).resolve()
    dates_df = pd.read_csv(dates_file, encoding="utf-8")

    # Get merged csv file paths
    merged_files = [f.path for f in os.scandir(args.folder) if f.is_file() and f.name.endswith(".csv")]
    #extract study ids from file names
    study_ids = set([os.path.basename(file).split("_")[0] + "_" + os.path.basename(file).split("_")[1] for file in merged_files])
    trials =['V1', 'V2']


    print(f"\n{'='*20}")
    print("\nStart Uploading Data to REDCap")
    for study_id in study_ids:
        print(f"\n{'-'*20}")
        print(f"\nProcessing files for {study_id}")
        for trial in trials:
            print(f"\nProcessing files for trial {trial}...")
            event_name = "baseline_period_arm_1" if trial == "V1" else "intervention_perio_arm_1"
            #get csv files for study_id and trial
            csv_files = [file for file in merged_files if study_id in file and trial in file]
            #upload csv files to REDCap using API
            for file in csv_files:
                #map file name to field name
                file_name = os.path.basename(file).split("_")[3]
                field_name = filed_mapping[filed_mapping["file_name"] == file_name]["field_name"].values[0] if not filed_mapping[filed_mapping["file_name"] == file_name].empty else np.nan
                if field_name is np.nan:
                    print(f"Warning: No field name mapping found for file {file_name}. Skipping upload.")
                    print(f"\n{'-'*20}")
                    continue
                #upload file to REDCap using API
                #pass file not path to api_upload function
                status_code = api_file_upload(file, field_name = field_name, study_id = study_id, event_name = event_name)
                if status_code == 200:
                    #move file to new folder "done" after successful upload
                    done_folder = folder / "done"
                    done_folder.mkdir(exist_ok=True)
                    os.rename(file, done_folder / os.path.basename(file))
            
            #*upload start and end date
            print("Upload Start and End Date")
            #get min and max date from date extraction result file data_dates_extraction_yyyymmdd.csv
            dates_extraction = dates_df[(dates_df["study_id"] == study_id) & (dates_df["visit_name"] == trial)]
            if dates_extraction.empty:
                print(f"Warning: No dates found for study ID {study_id}. Skipping date upload.")
                print(f"\n{'-'*20}")
                continue
            dates_extraction = dates_extraction.copy()
            dates_extraction["start_date_time"] = pd.to_datetime(dates_extraction["start_date_time"], format='%d-%m-%Y %H:%M')
            dates_extraction["end_date_time"] = pd.to_datetime(dates_extraction["end_date_time"], format='%d-%m-%Y %H:%M')

            start_dt = min(dates_extraction["start_date_time"]).strftime('%Y-%m-%d %H:%M')
            end_dt = max(dates_extraction["end_date_time"]).strftime('%Y-%m-%d %H:%M')
            print(f"Start date: {start_dt}, End date: {end_dt}")
            api_date_upload(study_id=study_id, event_name=event_name, start_date=start_dt, end_date=end_dt)


        print(f"\n{'-'*20}")
        print(f"CSV files and dates uploaded successfully to REDCap for study ID {study_id}")
        print(f"\n{'-'*20}")


    total_elapsed_time = time.time() - total_start_time
    print(f"\nTotal elapsed time: {total_elapsed_time:.2f} seconds")
    print("\nEnd Uploading Data to REDCap")
    print(f"\n{'='*20}")


if __name__ == "__main__":
    upload_to_redcap()

