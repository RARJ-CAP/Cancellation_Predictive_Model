import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import OneHotEncoder

# Setup logging to track ETL steps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure reproducibility
fake = Faker()
np.random.seed(42)
random.seed(42)

def generate_synthetic_data(n_samples):
    """
    Simulates the extraction of reservation data by generating synthetic records.
    Each record includes reservation details, guest behavior, and operational context.
    """
    data = []
    for _ in range(n_samples):
        reservation_datetime = fake.date_time_between(start_date='-30d', end_date='+30d')
        booking_channel = random.choice(['web', 'phone', 'kiosk', 'app', 'other'])
        lead_time_days = random.randint(0, 30)
        party_size = random.randint(1, 10)
        no_show_rate = round(random.uniform(0, 1), 2)
        visit_frequency = random.randint(0, 20)
        table_assigned = random.choice([True, False])
        responded_to_confirmation = random.choice([True, False])
        occupancy_rate = round(random.uniform(0.3, 1.0), 2)

        will_cancel = int((no_show_rate > 0.5 and not responded_to_confirmation) or random.random() < 0.1)

        data.append([
            reservation_datetime, booking_channel, lead_time_days, party_size,
            no_show_rate, visit_frequency, table_assigned, responded_to_confirmation,
            occupancy_rate, will_cancel
        ])

    columns = [
        'reservation_datetime', 'booking_channel', 'lead_time_days', 'party_size',
        'no_show_rate', 'visit_frequency', 'table_assigned', 'responded_to_confirmation',
        'occupancy_rate', 'will_cancel'
    ]
    return pd.DataFrame(data, columns=columns)

def transform_data(df):
    """
    Transforms raw data into a machine learning-ready format.
    Includes feature engineering and encoding of categorical variables.
    """
    logging.info("Transforming data...")

    # Extract hour and day of week from datetime
    df['reservation_hour'] = df['reservation_datetime'].dt.hour
    df['reservation_dayofweek'] = df['reservation_datetime'].dt.dayofweek

    # Drop the original datetime column
    df.drop(columns=['reservation_datetime'], inplace=True)

    # One-hot encode the booking channel (drop first to avoid multicollinearity)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded = encoder.fit_transform(df[['booking_channel']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['booking_channel']))

    # Combine encoded columns with the rest of the dataset
    df = pd.concat([df.drop(columns=['booking_channel']), encoded_df], axis=1)

    return df

def main():
    n_samples = 10000
    df = generate_synthetic_data(n_samples)
    df = transform_data(df)

    # Preview the cleaned dataset
    print(df.head())
    df.to_excel("Output.xlsx", index=False)
    print("\nData types:\n", df.dtypes)
    print("\nClass balance (will_cancel):\n", df['will_cancel'].value_counts(normalize=True))

if __name__ == "__main__":
    main()

