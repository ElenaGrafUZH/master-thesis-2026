import argparse
import os
import sys
import zipfile
import time
from collections import defaultdict
from pathlib import Path
import pandas as pd


def main():
    """
    Folder structure:
    - DEC_XX/
        - Data_Exports/
            - DEC_XX_V1_Wearables_Export/
                - DEC_XX_V1_Labfront_yyymmdd.zip
                    - DEC-XX_yyyyyyyy/
                        - garmin-*/
                            - *.csv
            - DEC_XX_V2_Wearables_Export/
                - DEC_XX_V2_Labfront_yyymmdd.zip
                    - DEC-XX_yyyyyyyy/
                        - garmin-*/
                            - *.csv
    Goal: Merge all CSVs of one garmin-* subfolder of a folder into one CSV file 
    named DEC-XX_V1_garmin-*_merged.csv and store it into designated output folder. 
    Skip non-garmin subfolders.
    """

    total_start_time = time.time()
    today_str = pd.Timestamp.today().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description="Merge all CSVs of one subfolder of a folder into one.")
    parser.add_argument("--folder", required=True, help="Folder containing DEC_XX folders (path to Main_Study)")
    parser.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    parser.add_argument("--output", default=None, help="Output folder path (default: same as script folder)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent          # .../scripts
    project_root = script_dir.parent                      # .../ (project root)

        
    if args.output is None:
        args_output = "output"
    else:
        args_output = args.output

    output_folder = Path(args_output)

    if not output_folder.is_absolute():
        output_folder = (project_root / output_folder).resolve()

    #check if output folder exists, if not create it (location of output folder is one step back from )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    

    dec_folders = [f.path for f in os.scandir(args.folder) if f.is_dir()]

    if not dec_folders:
        print(f"\nNo DEC_XX folders found in folder: {args.folder}")
        sys.exit(1)

    counter = 1
    len_dec_folders = len(dec_folders)
    #loop through each DEC_XX folder
    for dec in dec_folders:
        print(f"\n{'='*40}")
        print(f"\nProcessing DEC folder: {dec} ({counter}/{len_dec_folders})")
        dec_path = Path(dec)
        dec_string = dec_path.name
        data_exports = os.path.join(dec, "Data_Exports")
        if not os.path.exists(data_exports):
            print(f"No Data_Exports folder found in {dec}. Skipping.")
            continue

        for trial in ["V1", "V2"]:
            v_data = os.path.join(data_exports, f"{dec_string}_{trial}_Wearables_Export")
            print(f"\n{'-'*30}")
            print(f"\nProcessing {v_data}")
            if not os.path.exists(v_data):
                print(f"\nNo {v_data} found in {data_exports}. Skipping.")
                continue

            #open zip folder and look for subfolders
            zip_path = [f.path for f in os.scandir(v_data) if f.is_file() and f.name.endswith(".zip")]

            if len(zip_path)== 0:
                print(f"\nNo zip file found in {v_data}. Skipping.")
                continue
            if len(zip_path) > 1:
                #if more han one zip file found use the one with "Corrected" in name
                zip_path_corrected = [f for f in zip_path if "Corrected" in f]
                if len(zip_path_corrected) == 1:
                    zip_path = zip_path_corrected
                else:
                    print(f"\nMultiple zip files found in {v_data} and no unique one with 'Corrected' in name. Skipping.")
                    continue
            with zipfile.ZipFile(zip_path[0], 'r') as z: 
                #get last part of zip file path
                print(f"Processing zip file: {zip_path[0].split('/')[-1]}")
                # Single pass: group CSVs by garmin subfolder
                subfolder_files = defaultdict(list)
                for f in z.namelist():
                    if f.endswith(".csv"):
                        parent = f.rsplit("/", 1)[0]
                        if "garmin" in parent:
                            subfolder_files[parent].append(f)

                for sf, files in subfolder_files.items():
                    if sf.startswith("__MACOSX") or "garmin" not in sf:
                        print(f"Skipping non-garmin or system file subfolder: {sf}")
                        continue
                    print(f"\n{'*'*20}")
                    print(f"Processing subfolder: {sf}")
                    garmin_name = sf.rsplit("/", 1)[-1]
                    output = Path(output_folder) / f"{dec_string}_{trial}_{garmin_name}_merged_{today_str}.csv"
  

                    print(f"Found {len(files)} CSV files. Merging into {output}...")

                    dfs = []
                    for csv in files:
                       
                        with z.open(csv) as f:
                            try:
                                df = pd.read_csv(f, encoding=args.encoding, skiprows=6)
                                if not df.empty:
                                    dfs.append(df)
                                    
                            except pd.errors.EmptyDataError:
                                print(f"Skipping empty file: {csv}")

   
                    if not dfs:
                        print("No valid CSV files to merge.")
                    else:
                        merged = pd.concat(dfs, ignore_index=True)
                        merged['study_id'] = dec_string
                        merged['visit'] = trial
                        merged.to_csv(output, index=False, encoding=args.encoding)

                        print(f"Merging done for subfolder: {sf}, files merged: {len(dfs)}, \noutput file: {output}")


                print("\nDone!")
                print(f"All Files merged of folder: {v_data}")
        
        print(f"\nEnd of DEC folder: {dec}")
        print(f"\n{'='*40}")
        counter += 1
    total_elapsed_time = time.time() - total_start_time
    print(f"\nTotal elapsed time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()