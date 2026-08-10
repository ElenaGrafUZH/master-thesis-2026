import pytz
import requests
import pandas as pd
import configparser
import numpy as np

config = configparser.ConfigParser()
#! move back to "../../settings.ini" for .ipynb files
config.read("../../settings.ini")

def clean_data_baseline_optimized(dfs):
    """Cut Data to match visit dates of each study ID"""
    #1. create dataframe with baseline visit dates for each study ID
    ## get visit dates from REDCap
    data = {
        'token': config['REDCap']['api_token'],
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
    #print only if errer
    if r.status_code != 200:
        print('HTTP Status: ' + str(r.status_code))

    df_visit_dates= pd.DataFrame(r.json())


    #2. Prepare baseline visit dates
    ###create column with visit name only
    df_visit_dates['visit_name'] = df_visit_dates['redcap_event_name'].str.split(' ').str[0]

    ###df only with baseline information
    df_baseline_dates = df_visit_dates[(df_visit_dates['visit_name'] == 'Visit_1a')| (df_visit_dates['visit_name'] == 'Visit_1b')][['study_id', 'visit_name', 'visit_date']].copy()
    df_baseline_dates['visit_date'] = pd.to_datetime(
        df_baseline_dates['visit_date'],
        errors='coerce'
    )
    # Convert long format to wide format:
    # one row per study_id, columns Visit_1a and Visit_1b
    df_visit_timerange = (
        df_baseline_dates
        .pivot_table(
            index='study_id',
            columns='visit_name',
            values='visit_date',
            aggfunc='first'
        )
        .reset_index()
    )
    # If Visit_1b exists, baseline_end = Visit_1b
    # Otherwise baseline_end = Visit_1a
    df_visit_timerange['baseline_end'] = (
        df_visit_timerange['Visit_1b']
        .fillna(df_visit_timerange['Visit_1a'])
    )

    # baseline_start is always 14 days before Visit_1a
    df_visit_timerange['baseline_start'] = (
        df_visit_timerange['Visit_1a'] - pd.Timedelta(days=14)
    )

    # Include the whole baseline_end day by adding one day,
    # then later filtering with < baseline_end
    df_visit_timerange['baseline_end'] = (
        df_visit_timerange['baseline_end'] + pd.Timedelta(days=1)
    )

    # Localize Zurich time and convert to UTC
    zurich_tz = pytz.timezone("Europe/Zurich")

    df_visit_timerange['baseline_start'] = (
        df_visit_timerange['baseline_start']
        .dt.tz_localize(zurich_tz)
        .dt.tz_convert("UTC")
    )

    df_visit_timerange['baseline_end'] = (
        df_visit_timerange['baseline_end']
        .dt.tz_localize(zurich_tz)
        .dt.tz_convert("UTC")
    )

    df_visit_timerange = df_visit_timerange[
        ['study_id', 'baseline_start', 'baseline_end']
    ]

    # 3. go thorough all dfs and drop rows that do not lie within baseline visit time range
    start_map = df_visit_timerange.set_index('study_id')['baseline_start']
    end_map = df_visit_timerange.set_index('study_id')['baseline_end']

    cleaned_dfs = []
    
    for df in dfs:
        df = df.copy()

        datetime_utc_parsed = pd.to_datetime(
            df['datetime_utc'],
            utc=True,
            format='mixed',
            errors='coerce'
        )

        baseline_start = df['study_id'].map(start_map)
        baseline_end = df['study_id'].map(end_map)

        mask = (
            (datetime_utc_parsed >= baseline_start) &
            (datetime_utc_parsed < baseline_end)
        )

        cleaned_dfs.append(df.loc[mask].copy())

    return cleaned_dfs


def clean_data_no_data_days_optimized(dfs):
    """Drop days where no data was collected"""
    #! move back to "../../data/checks/df_daily_summary_no_data.csv" for .ipynb files
    df_no_data = pd.read_csv("../../data/checks/df_daily_summary_no_data_2026-07-08.csv")
    # Normalise date format in df_no_data once before the loop
    df_no_data["date_no_data"] = pd.to_datetime(
        df_no_data["date_no_data"],
        errors="coerce"
    ).dt.date

    # Build a set of (study_id, date) tuples for O(1) lookup instead of merging
    no_data_set = set(zip(df_no_data["study_id"], df_no_data["date_no_data"]))

    #2. go through all files and drop rows with study id and date of no-data-days
    #2.1 report how many rows were dropped for each file and how many rows remain after dropping
    cleaned_dfs = []
    for i, df in enumerate(dfs):
        df = df.copy()

        # Parse UTC once
        dt_utc = pd.to_datetime(
            df["datetime_utc"],
            errors="coerce",
            utc=True
        )

        # Convert each row to its own local timezone, then extract local date
        df["date"] = [
            ts.tz_convert(tz).date() if pd.notna(ts) and pd.notna(tz) else pd.NaT
            for ts, tz in zip(dt_utc, df["timezone"])
        ]

        before_rows = df.shape[0]
        
        # Vectorised mask using set lookup instead of merge
        mask = ~pd.Series(
            zip(df["study_id"], df["date"]), index=df.index
        ).isin(no_data_set)

        df = df.loc[mask].drop(columns=["date"])
        after_rows = df.shape[0]
        
        # Print report per dataframe
        print(f"Dataframe {i+1}: dropped {before_rows - after_rows} rows due to no-data-days. Remaining rows: {after_rows}")
        
        cleaned_dfs.append(df)

    return cleaned_dfs


def drop_sleep_window_data(df, df_sleep, id_col='study_id', ts_col='datetime', start_col='onset_time', end_col='wake_up_time', tz_col ='timezone'):
    
    df = df.copy()
    df_sleep = df_sleep.copy()

    dt_utc = pd.to_datetime(
                df["datetime_utc"],
                errors="coerce",
                utc=True
            )

    df[ts_col] = [
        ts.tz_convert(tz).replace(tzinfo=None) if pd.notna(ts) and pd.notna(tz) else pd.NaT
        for ts, tz in zip(dt_utc, df[tz_col])
    ]

    for col in [start_col, end_col]:
            dt = pd.to_datetime(df_sleep[col], errors="coerce", utc=True)
            df_sleep[col] = [
                ts.tz_convert(tz).replace(tzinfo=None) if pd.notna(ts) and pd.notna(tz) else pd.NaT
                for ts, tz in zip(dt, df_sleep[tz_col])
            ]

    keep_mask = pd.Series(True, index = df.index)

    for sid, windows in df_sleep.groupby(id_col):
        sid_mask = df[id_col] == sid
        ts = df.loc[sid_mask, ts_col].values

        starts = windows[start_col].values
        ends = windows[end_col].values

        in_any_window = np.any(
            (ts[:, None] >= starts[None, :]) & (ts[:, None] <= ends[None, :]), 
            axis=1
        )
        keep_mask.loc[sid_mask] = ~in_any_window

    return df[keep_mask].reset_index(drop=True)


def keep_timespan_only(df, df_times, start_col, end_col, window_col, id_col='study_id', ts_col='datetime', tz_col='timezone'):
    df = df.copy()
    dt_utc_d = pd.to_datetime(
                df["datetime_utc"],
                errors="coerce",
                utc=True
            )

    df[ts_col] = [
        ts.tz_convert(tz).replace(tzinfo=None) if pd.notna(ts) and pd.notna(tz) else pd.NaT
        for ts, tz in zip(dt_utc_d, df[tz_col])
    ]


    keep_mask = pd.Series(True, index = df.index)

    for sid, windows in df_times.groupby(id_col):
        sid_mask = df[id_col] == sid
        ts = df.loc[sid_mask, ts_col].values

        starts = windows[start_col].apply(
                 lambda x: x.replace(tzinfo=None) if pd.notna(x) else pd.NaT
             ).values
        ends = windows[end_col].apply(
                 lambda x: x.replace(tzinfo=None) if pd.notna(x) else pd.NaT
             ).values
        window_nums = windows[window_col].values

        match_matrix = (ts[:, None] >= starts[None, :]) & (ts[:, None] <= ends[None, :])

        match_idx = np.where(match_matrix.any(axis=1), match_matrix.argmax(axis=1), -1)

        in_any_window = match_idx != -1
        keep_mask.loc[sid_mask] = in_any_window

        # Assign window number only for matched rows
        matched_positions = df.index[sid_mask][in_any_window]
        df.loc[matched_positions, window_col] = window_nums[match_idx[in_any_window]]


    return df[keep_mask].reset_index(drop=True)