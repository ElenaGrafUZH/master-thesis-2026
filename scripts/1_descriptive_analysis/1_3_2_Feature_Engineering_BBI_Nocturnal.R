# ============================================================
#  HRV Analysis Pipeline — Garmin Beat-to-Beat Data
#  Prerequisites: install.packages(c("RHRV", "lubridate", "dplyr", "ggplot2"))
# ============================================================

library(RHRV)
library(lubridate)
library(dplyr)
library(ggplot2)
library(ini)
library(jsonlite)
library(httr)
library(stringr)
library(tibble)

config <- read.ini("settings.ini")

# ============================================================
# GET VISITDATES FROM csv
# ============================================================
df_visit_timerange <- read.csv("data/checks/visit_dates_2026-07-08.csv")

df_visit_timerange <- df_visit_timerange %>%
  filter(visit_name == "V1") %>%
  rename(baseline_start = start_visit_date, baseline_end = end_visit_date) %>%
  select(-visit_name) %>%
  mutate(
    baseline_start = as.Date(baseline_start, , format = "%Y-%m-%d"),
    baseline_end   = as.Date(baseline_end, , format = "%Y-%m-%d")
  )

# ============================================================
# GET SLEEP WINDOWS FROM csv
# ============================================================

df_sleep_windows <- read.csv("data/checks/data_sleep_windows_extraction_2026-07-08.csv")

# ============================================================
# GET NO DATA DAYS FROM csv
# ============================================================
df_no_data_days <- read.csv("data/checks/df_daily_summary_no_data_2026-07-08.csv")
df_no_data_days <- df_no_data_days %>%
  mutate(
    date_no_data = as.Date(date_no_data, tz = "Europe/Zurich")
  )

# ============================================================
# FUNCITON: run RHRV on one session
# ============================================================

run_hrv_session <- function(bbi_ms, power_shift = 30) {
  # Need at least ~2 minutes of data for meaningful HRV
  if (length(bbi_ms) < 60) {
    cat("\n Session too short, skipping.")
    return(NULL)
  }

  # Convert bbi from milliseconds to seconds (RHRV requirement)
  bbi_sec <- bbi_ms / 1000
  beat_times <- cumsum(bbi_sec) # cumulative time axis starting at 0

  # Build RHRV data structure
  hrv <- CreateHRVData()
  hrv <- SetVerbose(hrv, FALSE) # set to TRUE if you want diagnostic output
  hrv$Beat <- data.frame(Time = beat_times)

  # Build instantaneous HR signal
  hrv <- BuildNIHR(hrv)

  # Filter outlier beats (ectopics, noise)
  hrv <- FilterNIHR(hrv)

  # Interpolate to uniform 4 Hz signal (needed for frequency analysis)
  hrv <- InterpolateNIHR(hrv, freqhr = 4)

  # --- Time-domain metrics (SDNN, RMSSD, pNN50, mean HR etc.) ---
  # 'size' = window size in seconds (300s = 5 min, standard in HRV literature)
  hrv <- CreateTimeAnalysis(hrv, size = 300, interval = 7.8125)

  # --- Frequency-domain (LF, HF bands) ---
  hrv <- CreateFreqAnalysis(hrv)

  # Only run frequency analysis if session is long enough (need >5 min)
  total_sec <- max(beat_times)
  if (total_sec > 300) {
    hrv <- CalculatePowerBand(
      hrv,
      indexFreqAnalysis = 1,
      size = 300, # window: 5 min
      shift = power_shift, # step: 30 seconds
      type = "fourier"
    )
  }

  return(hrv)
}
# ============================================================
# LOAD DATA
# ============================================================
# Get all bbi files
folder <- "data/bbi_device"
cat("Searching for bbi files in data/bbi...")
bbi_files <- list.files(
  path = folder,
  pattern = "garmin-device-bbi.*\\.csv$",
  full.names = TRUE
)


bbi_files <- tibble(file_path = bbi_files) %>%
  mutate(
    file_name = basename(file_path),
    study_id = str_extract(file_name, "DEC_\\d+|DC_\\d+")
  ) %>%
  pull(file_path)


cat("\nFound", length(bbi_files), "bbi files")

# loop through all files and concatenate only in the end the results
all_daily_summary <- data.frame()
all_two_week_summary <- data.frame()
all_session_summary <- data.frame()
all_df <- data.frame()
all_fractured_night <- data.frame()
counter <- 1

for (bbi_file in bbi_files) {
  cat("\n", strrep("=", 40), "\n", sep = "")
  cat("(", counter, "/", length(bbi_files), ") Processing file:", bbi_file)
  # ============================================================
  # 1. CLEAN DATA
  # ============================================================
  # read file
  df_initial <- read.csv(bbi_file, stringsAsFactors = FALSE)
  # extract study id
  dec_id <- unique(df_initial$study_id)

  # clean df
  cat("\n", strrep("-", 20), "\n", sep = "")
  cat("Cleaning dataframe...")
  cat("\nTotal beats before cleaning:", nrow(df_initial), "\n")


  # Parse timestamp, keeping the Europe/Zurich timezone
  df <- df_initial %>%
    mutate(
      datetime = ymd_hms(isoDate, tz = "Europe/Zurich"),
      date     = as.Date(datetime, tz = "Europe/Zurich")
    ) %>%
    # Sort chronologically (important before gap detection)
    arrange(datetime) %>%
    # Sanity check: drop physiologically impossible intervals
    # Normal human range: ~300ms (200 bpm) to ~2000ms (30 bpm)
    filter(bbi >= 300, bbi <= 2000)

  # Drop no data days (from daily summary)
  df <- df <- df %>%
    left_join(df_no_data_days, by = c("study_id", "date" = "date_no_data"), keep = TRUE) %>%
    filter(is.na(date_no_data)) %>%
    select(-date_no_data)
  df <- df %>%
    select(-study_id.y, study_id = study_id.x)

  cat("Total beats after cleaning:", nrow(df), "\n")
  cat("Date range:", format(min(df$date)), "to", format(max(df$date)))

  # ============================================================
  # 2. CUT DATA TO SLEEP WINDOWS
  # ============================================================

  # filter to only have sleep window of one DEC
  sleep <- df_sleep_windows %>% filter(study_id == dec_id)
  sleep$sleep_date <- as.Date(sleep$wake_up_time)
  # keep only rows where wake up is <= visit end and wake up >= visit start + 1
  sleep <- sleep %>%
    left_join(df_visit_timerange, by = "study_id") %>%
    filter(
      wake_up_time >= baseline_start + 1,
      wake_up_time <= baseline_end
    )

  # For each BBI point, check if it falls within any sleep window
  df <- do.call(rbind, lapply(seq_len(nrow(sleep)), function(i) {
    df %>%
      filter(datetime >= sleep$onset_time[i] & datetime <= sleep$wake_up_time[i]) %>%
      mutate(sleep_date = sleep$sleep_date[i])
  }))


  cat("\nTotal beats after cropping to sleep windows:", nrow(df), "\n")
  # ============================================================
  # 3. DETECT GAPS BETWEEN SESSIONS
  # ============================================================
  cat("\n", strrep("-", 20), "\n", sep = "")
  cat("Creating Sessions...")
  # Define a gap threshold
  GAP_THRESHOLD_SEC <- 150

  df <- df %>%
    mutate(
      # Time since previous beat in seconds
      time_diff_sec = as.numeric(difftime(datetime, lag(datetime), units = "secs")),
      # First row has no previous beat, set to 0
      time_diff_sec = ifelse(is.na(time_diff_sec), 0, time_diff_sec),
      # Flag start of a new session
      new_session = time_diff_sec > GAP_THRESHOLD_SEC,
      # Assign a session ID (cumulative sum of new session flags + 1)
      session_id = cumsum(time_diff_sec > GAP_THRESHOLD_SEC) + 1
    )

  all_df <- rbind(all_df, df)

  # Quick summary of sessions found
  session_summary <- df %>%
    group_by(session_id, study_id) %>%
    summarise(
      sleep_date = unique(sleep_date),
      start_time = min(datetime),
      end_time = max(datetime),
      n_beats = n(),
      duration_min = as.numeric(difftime(max(datetime), min(datetime), units = "mins")),
      duration_sec = as.numeric(difftime(max(datetime), min(datetime), units = "secs")),
      .groups = "drop"
    )

  all_session_summary <- rbind(all_session_summary, session_summary)


  # keep only sessions that cover at least 80% of the known sleep duration
  sleep$durationInMin <- sleep$durationInMs / 60000

  # Join sleep duration to session summary
  session_summary <- session_summary %>%
    left_join(sleep %>% select(sleep_date, durationInMin), by = "sleep_date")

  # Filter sessions covering at least 80% of known sleep
  valid_sessions <- session_summary %>%
    filter(duration_min >= 0.8 * durationInMin) %>%
    pull(session_id)

  # only keep valid sessions in bbi data
  df <- df %>%
    filter(session_id %in% valid_sessions)

  # ============================================================
  # 4. LOOP OVER SESSIONS — collect metrics
  # ============================================================
  cat("\n", strrep("-", 20), "\n", sep = "")
  cat("Calculating HRV values with RHRV...")
  results <- list()

  for (sid in unique(df$session_id)) {
    session_data <- df %>% filter(session_id == sid)
    cat("\n Running RHRV for", dec_id, "session", sid)
    hrv <- run_hrv_session(session_data$bbi, power_shift = 30)

    if (!is.null(hrv)) {
      # Pull out the time-domain summary
      td <- hrv$TimeAnalysis[[1]]
      fd <- hrv$FreqAnalysis[[1]]

      results[[length(results) + 1]] <- data.frame(
        session_id = sid,
        study_id = session_data$study_id,
        date = max(session_data$date),
        start_time = min(session_data$datetime),
        end_time = max(session_data$datetime),
        n_beats = nrow(session_data),
        sdnn = td$SDNN, # overall HRV
        rmssd = td$rMSSD, # short-term HRV (parasympathetic)
        pnn50 = td$pNN50, # % of successive beats differing >50ms
        # Frequency domain — log-transformed band power
        mean_logHF = mean(log(fd$HF[fd$HF > 0]), na.rm = TRUE),
        mean_logLF = mean(log(fd$LF), na.rm = TRUE),
        mean_logVLF = mean(log(fd$VLF), na.rm = TRUE),
        mean_LFHF = mean(fd$LFHF, na.rm = TRUE),
        median_logHF = median(log(fd$HF[fd$HF > 0]), na.rm = TRUE),
        median_logLF = median(log(fd$LF), na.rm = TRUE),
        median_logVLF = median(log(fd$VLF), na.rm = TRUE),
        median_LFHF = median(fd$LFHF, na.rm = TRUE),
        std_logHF = sd(log(fd$HF), na.rm = TRUE),
        std_logLF = sd(log(fd$LF), na.rm = TRUE),
        std_LFHF = sd(fd$LFHF, na.rm = TRUE),
        stringsAsFactors = FALSE
      )
    }
  }

  session_results <- bind_rows(results)
  session_results <- unique(session_results)
  # if one night was fractured keep only the longest session
  session_results_all <- session_results

  session_results <- session_results %>%
    group_by(date) %>%
    slice_max(order_by = n_beats, n = 1, with_ties = FALSE) %>%
    ungroup()

  fractured_nights <- session_results_all %>%
    anti_join(session_results, by = names(session_results_all))
  all_fractured_night <- rbind(all_fractured_night, fractured_nights)
  cat("\nDone.")

  # filter those sessions out from df
  df <- df %>%
    anti_join(fractured_nights %>% select(session_id), by = "session_id")

  all_session_summary <- all_session_summary %>%
    anti_join(fractured_nights %>% select(session_id), by = "session_id")


  # ============================================================
  # 5. AGGREGATE TO DAILY & OVERALL SUMMARIES
  # ============================================================
  cat("\n", strrep("-", 20), "\n", sep = "")
  cat("Calculating Daily and 2-Weekly summary...")
  # Daily averages (mean across all sessions within a day)
  daily_summary <- session_results %>%
    group_by(date, study_id) %>%
    summarise(
      # daily_mean_hr   = mean(mean_hr,  na.rm = TRUE),
      # daily_mean_hr_new = mean(rate_of_means_hr, na.rm=TRUE),
      session_id = session_id[which.max(n_beats)],
      daily_mean_sdnn = mean(sdnn, na.rm = TRUE),
      daily_median_sdnn = median(sdnn, na.rm = TRUE),
      daily_mean_rmssd = mean(rmssd, na.rm = TRUE),
      daily_median_rmssd = median(rmssd, na.rm = TRUE),
      daily_mean_pnn50 = mean(pnn50, na.rm = TRUE),
      daily_median_pnn50 = median(pnn50, na.rm = TRUE),
      daily_mean_logHF = mean(mean_logHF, na.rm = TRUE),
      daily_median_logHF = median(median_logHF, na.rm = TRUE),
      daily_mean_logLF = mean(mean_logLF, na.rm = TRUE),
      daily_median_logLF = median(median_logLF, na.rm = TRUE),
      daily_mean_logVLF = mean(mean_logVLF, na.rm = TRUE),
      daily_median_logVLF = median(median_logVLF, na.rm = TRUE),
      daily_mean_LFHF = mean(mean_LFHF, na.rm = TRUE),
      daily_median_LFHF = median(median_LFHF, na.rm = TRUE),
      n_sessions = n(),
      .groups = "drop"
    )

  daily_hr <- df %>%
    group_by(session_id, study_id) %>%
    summarise(
      # daily_mean_hr_new = mean(60000/bbi, na.rm = TRUE),
      daily_mean_hr = 60000 / mean(bbi, na.rm = TRUE),
      daily_median_hr = 60000 / median(bbi, na.rm = TRUE),
      n_beats = n(),
      .groups = "drop"
    )
  daily_summary <- daily_summary %>%
    left_join(daily_hr, by = c("session_id", "study_id"))

  # Overall 2-week averages

  two_week_summary <- daily_summary %>%
    group_by(study_id) %>%
    summarise(
      mean_hr = mean(daily_mean_hr, na.rm = TRUE),
      median_hr = median(daily_median_hr, na.rm = TRUE),
      mean_sdnn = mean(daily_mean_sdnn, na.rm = TRUE),
      median_sdnn = median(daily_median_sdnn, na.rm = TRUE),
      mean_rmssd = mean(daily_mean_rmssd, na.rm = TRUE),
      median_rmssd = median(daily_median_rmssd, na.rm = TRUE),
      mean_pnn50 = mean(daily_mean_pnn50, na.rm = TRUE),
      median_pnn50 = median(daily_median_pnn50, na.rm = TRUE),
      mean_logHF = mean(daily_mean_logHF, na.rm = TRUE),
      median_logHF = median(daily_median_logHF, na.rm = TRUE),
      mean_logLF = mean(daily_mean_logLF, na.rm = TRUE),
      median_logLF = median(daily_median_logLF, na.rm = TRUE),
      mean_logVLF = mean(daily_mean_logVLF, na.rm = TRUE),
      median_logVLF = median(daily_median_logVLF, na.rm = TRUE),
      mean_LFHF = mean(daily_mean_LFHF, na.rm = TRUE),
      median_LFHF = median(daily_median_LFHF, na.rm = TRUE),
      n_nights = n(),
      .groups = "drop"
    )

  # Append to accumulators
  all_daily_summary <- rbind(all_daily_summary, daily_summary)
  all_two_week_summary <- rbind(all_two_week_summary, two_week_summary)
  cat("\nDone.")

  cat("\nDone Processing file:", bbi_file)
  counter <- counter + 1
}

# ============================================================
# 6. STORE AS CSV FILES
# ============================================================
cat("\n", strrep("=", 40), "\n", sep = "")
cat("Storing Summaries to output/1_feature_extraction...")
write.csv(all_daily_summary, "output/1_feature_extraction/df_features_daily_nocturnal_hr_2026-07-08.csv", row.names = FALSE)

write.csv(all_two_week_summary, "output/1_feature_extraction/df_features_nocturnal_hr_2026-07-08.csv", row.names = FALSE)

cat("\nDone!")
cat("\n", strrep("=", 40), "\n", sep = "")
