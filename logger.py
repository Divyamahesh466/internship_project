import csv
import os
from datetime import datetime

LOG_FILE = "logs/attendance_log.csv"

os.makedirs("logs", exist_ok=True)


def initialize_log():

    if not os.path.exists(LOG_FILE):

        with open(
            LOG_FILE,
            mode="w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Timestamp",
                "Person_Name",
                "State"
            ])


def log_event(
    person_name,
    state
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        LOG_FILE,
        mode="a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            person_name,
            state
        ])

    print(
        f"[LOG] {timestamp} | "
        f"{person_name} | "
        f"{state}"
    )
    