import pandas as pd
import os
from datetime import datetime

# The name of the file where we will store logs
LOG_FILE = "monitoring_logs.csv"

def log_prediction(model_version, input_data, prediction, latency, feedback):
    """
    Logs the details of a prediction to a CSV file.
    """
    # Create a dictionary for the new log entry
    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_version': model_version,
        'bill_length': input_data[0],
        'bill_depth': input_data[1],
        'flipper_length': input_data[2],
        'body_mass': input_data[3],
        'prediction': prediction,
        'latency': latency,
        'user_feedback': feedback
    }

    # Convert dictionary to DataFrame
    df_log = pd.DataFrame([log_entry])

    # Append to CSV (create if it doesn't exist, otherwise append without header)
    if not os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, index=False)
    else:
        df_log.to_csv(LOG_FILE, mode='a', header=False, index=False)