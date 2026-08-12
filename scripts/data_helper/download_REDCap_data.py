import requests
import os
from pathlib import Path
from email.message import Message

import configparser

config = configparser.ConfigParser()

script_dir = Path(__file__).resolve().parent      # project/scripts
project_root = script_dir.parent.parent                  # project

config_path = project_root / "settings.ini"

config.read(config_path)

# !when on insel network add proxy settings
# proxy = {
#     "http": 'http://enumbergoeshere:passwordgoeshere@proxy-prod.insel.ch:8080',
#     "https": 'http://enumbergoeshere:passwordgoeshere@proxy-prod.insel.ch:8080'
# }


  
#* =========================
#* 1. DOWNLOAD FILE
#* =========================
def download_file(record, field, event, output_dir):
    folder = Path(output_dir)
    if not folder.is_absolute():
        folder = (project_root / folder).resolve()

    folder.mkdir(parents=True, exist_ok=True)
    data = {
            'token': config['REDCap']['api_token'],
            'content': 'file',
            'action': 'export',
            'record': record,
            'field': field,
            'event': event,
            'returnFormat': 'json'
        }

    r = requests.post(config['REDCap']['api_url'], data=data, timeout=600)#, proxies=proxy)
  
    if r.status_code == 200 and len(r.content) > 0:

        content_type = r.headers.get('Content-Type', '')
        msg = Message()
        msg['Content-Type'] = content_type
        params = dict(msg.get_params()[1:])  # Skip the first parameter which is the main content type
        filename = params.get('name')
        filepath = os.path.join(folder, filename)

        with open(filepath, 'wb') as f:
            f.write(r.content)

        return filepath
    else:
        print(f"Failed download: {record} | {field}")
        return None

#* =========================
#* 2. GET RECORD IDS
#* =========================
def get_record_ids():
    data = {
        "token": config['REDCap']['api_token'],
        "content": "record",
        "format": "json",
        "type": "flat",
        "fields": ["study_id"],   # use your actual record ID field name
    }

    r = requests.post(config['REDCap']['api_url'], data=data, timeout=100)#,proxies=proxy)
    r.raise_for_status()

    raw_records = r.json()

    record_ids = [rec["study_id"] for rec in raw_records]
    return set(record_ids)


#* =========================
#* 3. DOWNLOAD FILES FOR ALL RECORDS
#* =========================
def download_files_for_records(record_ids, field, event, output_dir):
    if record_ids[0] == 'all':
        print("Downloading files for all records...")
        record_ids = get_record_ids()
    else:
        print(f"Downloading files for specified records: {record_ids}")
    file_paths = []
    for record in record_ids:
        print(f"\n{'-'*20}")
        print(f"Downloading file for record: {record}...")
        path = download_file(record, field, event, output_dir)
        if path:
            file_paths.append(path)
        print(f"Finished downloading file for record: {record}")

    return file_paths
