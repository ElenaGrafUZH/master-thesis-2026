
import argparse
import os
import time
import requests
import pandas as pd
from pathlib import Path


def date_range_testing():
    """
    Compare dates from Visits with dates from downloaded Labfront data to check if they match. 
    This is a test function to check if there was any manual data collection error.
    """
    total_start_time = time.time()
    parser = argparse.ArgumentParser(description="Test Date Range Extraction")
    parser.add_argument("--data_dates", required=True, help="Path to the file containing extracted start and end dates (output/checks/data_dates_extraction.csv)")
    parser.add_argument("-o", "--output", help="Path to the output folder for log (default: output/checks)", default="output/checks")
    parser.add_argument("--redcap_token", required=True, help="REDCap API token")
    # parser.add_argument("--E-number", required=True, help="E-number")
    # parser.add_argument("--password", required=True, help="Password")
    
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent          # .../scripts
    project_root = script_dir.parent                      # .../ (project root)

    folder = Path(args.data_dates)
    if not folder.is_absolute():
        folder = (project_root / folder).resolve()

    # Check if output folder exists, if not create it
    output_folder = Path(args.output)
    if not output_folder.is_absolute():
        output_folder = (project_root / output_folder).resolve()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get merged csv file paths
    df_date_extraction = pd.read_csv(args.data_dates, encoding="utf-8")

    # Connect to REDCap API and get visit dates

    # !when on insel network add proxy settings
    # proxy = {
    #     "http": 'http://enumbergoeshere:passwordgoeshere@proxy-prod.insel.ch:8080',
    #     "https": 'http://enumbergoeshere:passwordgoeshere@proxy-prod.insel.ch:8080'
    # }


    data = {
        'token': args.redcap_token,
        'content': 'report',
        'format': 'json',
        'report_id': '263',
        'csvDelimiter': '',
        'rawOrLabel': 'label',
        'rawOrLabelHeaders': 'label',
        'exportCheckboxLabel': 'false',
        'returnFormat': 'json'
    }
    r = requests.post('https://redcap.unibe.ch/api/',data=data, timeout=10) #, proxies=proxy)
    print('HTTP Status: ' + str(r.status_code))
  
    df_visit_dates= pd.DataFrame(r.json())

    #create column with visit name only
    df_visit_dates['visit_name'] = df_visit_dates['redcap_event_name'].str.split(' ').str[0]

    #get unique study ids from df_date_extraction
    study_ids = df_date_extraction['study_id'].unique().tolist()
    exclude_study_id = list()
    
    print(f"\n{'='*20}")
    print("\nStart Date Range Testing")
    print(f"\n{'-'*20}")
    print("\nCreate visit time ranges for each study_id...")
    #create df with visit start and end date for each study_id and trial
    df_visit_timerange = pd.DataFrame(columns=["study_id", "visit_name", "start_visit_date", "end_visit_date"])
    for study_id in study_ids:
        #get visit dates for study_id
        df_visit_dates_study_id = df_visit_dates[df_visit_dates['study_id'] == study_id].reset_index(drop=True)

        #if only one visit date skip
        if df_visit_dates_study_id['visit_name'].nunique() == 1:
            #baseline time range when only V1a: V1a visit date - 14 days to V1a visit date
            baseline_end = pd.to_datetime(df_visit_dates_study_id[df_visit_dates_study_id['visit_name'] == 'Visit_1a']['visit_date'].iloc[0]).date()
            baseline_start = baseline_end - pd.Timedelta(days=14)

            intervention_start = None
            intervention_end = None

            exclude_study_id.append(study_id)
            
        #check if 2 visits or 4
        elif df_visit_dates_study_id['visit_name'].nunique() == 2:
            #baseline time range when only V1a: V1a visit date - 14 days to V1a visit date
            baseline_end = pd.to_datetime(df_visit_dates_study_id[df_visit_dates_study_id['visit_name'] == 'Visit_1a']['visit_date'].iloc[0]).date()
            baseline_start = baseline_end - pd.Timedelta(days=14)

            #intervention period: V1a visit date + 1d to V2a visit date
            intervention_start = baseline_end + pd.Timedelta(days=1)
            intervention_end = pd.to_datetime(df_visit_dates_study_id[df_visit_dates_study_id['visit_name'] == 'Visit_2a']['visit_date'].iloc[0]).date()
            
        else:
            #baseline time range when V1a and V1b: V1a visit date - 14 days to V1b visit date
            baseline_end = pd.to_datetime(df_visit_dates_study_id[df_visit_dates_study_id['visit_name'] == 'Visit_1b']['visit_date'].iloc[0]).date()
            baseline_start = pd.to_datetime(df_visit_dates_study_id[df_visit_dates_study_id['visit_name'] == 'Visit_1a']['visit_date'].iloc[0]).date() - pd.Timedelta(days=14)

            #intervention period: V1a visit date + 1d to V2b visit date
            intervention_start = baseline_end + pd.Timedelta(days=1)
            intervention_end = pd.to_datetime(df_visit_dates_study_id[df_visit_dates_study_id['visit_name'] == 'Visit_2b']['visit_date'].iloc[0]).date()


        df_visit_timerange = pd.concat([df_visit_timerange, pd.DataFrame({
            "study_id": [study_id],
            "visit_name": ["V1"],
            "start_visit_date": [baseline_start],
            "end_visit_date": [baseline_end]
        })], ignore_index=True)

        df_visit_timerange = pd.concat([df_visit_timerange, pd.DataFrame({
            "study_id": [study_id],
            "visit_name": ["V2"],
            "start_visit_date": [intervention_start],
            "end_visit_date": [intervention_end]
        })], ignore_index=True)
    
    
    visit_dates_file = os.path.join(args.output, "visit_dates.csv")
    df_visit_timerange.to_csv(visit_dates_file, index=False)
    print(f"Successfully created visit time ranges for each study_id and saved to {visit_dates_file}")

    print(f"\n{'-'*20}")
    print("\nMerge visit time ranges with date extraction results...")
    #merge df_visit_timerange with df_date_extraction to get garmin features for each visit time range
    df_date_extraction = pd.merge(df_date_extraction, df_visit_timerange, on=["study_id", "visit_name"], how="left")

    print("Successfully merged visit time ranges with date extraction results...")

    print(f"\n{'-'*20}")
    print("\nCheck if data falls into visit timerange...")
    #check if dates fall into timerange
    df_date_extraction['in_visit_range'] = ((pd.to_datetime(df_date_extraction['start_date']).dt.date <= pd.to_datetime(df_date_extraction['start_visit_date']).dt.date) &
                                        (pd.to_datetime(df_date_extraction['end_date']).dt.date >= pd.to_datetime(df_date_extraction['end_visit_date']).dt.date))

    print("Done checking if data falls into visit timerange")

    print(f"\n{'-'*20}")
    print("\nCheck if V1 and V2 data overlap...")
    #exclude study ids with only one visit date
    study_ids = [x for x in study_ids if x not in set(exclude_study_id)]
    #Check if V1 and V2 data overlap
    final_df = df_date_extraction[df_date_extraction['study_id'].isin(exclude_study_id)].copy().reset_index(drop=True)
    for study_id in study_ids:
        #for V1 check if end_date overlaps with visit start date of V2
        #!allowed 1 day overlap between V1 data end and V2 visit start
        v2_visit_start = df_visit_timerange[(df_visit_timerange['study_id'] == study_id) & (df_visit_timerange['visit_name'] == 'V2')]['start_visit_date'].iloc[0]
        df_v1 = df_date_extraction[(df_date_extraction['study_id'] == study_id) & (df_date_extraction['visit_name'] == 'V1')].copy().reset_index(drop=True)
        df_v1['overlaps'] = pd.to_datetime(df_v1['end_date']).dt.date -pd.Timedelta(days=1) >= pd.to_datetime(v2_visit_start).date()
        
        #for V2 check if start_date overlaps with visit end date of V1
        #!allowed 1 day overlap between V2 data start and V1 visit end
        v1_visit_end = df_visit_timerange[(df_visit_timerange['study_id'] == study_id) & (df_visit_timerange['visit_name'] == 'V1')]['end_visit_date'].iloc[0]
        df_v2 = df_date_extraction[(df_date_extraction['study_id'] == study_id) & (df_date_extraction['visit_name'] == 'V2')].copy().reset_index(drop=True)
        df_v2['overlaps'] = pd.to_datetime(df_v2['start_date']).dt.date + pd.Timedelta(days=1) <= pd.to_datetime(v1_visit_end).date()
        
        final_df = pd.concat([final_df, df_v1, df_v2], ignore_index=True)
    
    print("\nDone checking if V1 and V2 data overlap")

    print(f"\n{'-'*20}")
    print("\nSave results to visit_timerange_check.csv...")
    output_file = os.path.join(args.output, "visit_timerange_check.csv")
    final_df.to_csv(output_file, index=False)
    print("File successfully saved to:", output_file)


    total_elapsed_time = time.time() - total_start_time
    print(f"\nTotal elapsed time: {total_elapsed_time:.2f} seconds")
    print("\nEnd Date Range Testing")
    print(f"\n{'='*20}")


if __name__ == "__main__":
    date_range_testing()