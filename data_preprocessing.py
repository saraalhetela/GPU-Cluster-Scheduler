"""data_preprocessing.py -- loads gpu_prices.csv / jobs.csv for the env."""

import pandas as pd
import config


def load_price_data(path: str = config.PRICE_PATH) -> dict:
    """Returns {hour(int): gpu_price(float)}."""
    df = pd.read_csv(path)
    return dict(zip(df["hour"].astype(int), df["gpu_price"].astype(float)))


def load_job_data(path: str = config.JOBS_PATH, split: bool = False, seed: int = 42):
    """
    Returns either the full list of job dicts, or a (train, test) tuple of
    job-dict lists split via config.TRAIN_RATIO.
    """
    df = pd.read_csv(path)
    jobs = df.to_dict(orient="records")

    if not split:
        return jobs

    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train = int(len(df_shuffled) * config.TRAIN_RATIO)
    train_jobs = df_shuffled.iloc[:n_train].to_dict(orient="records")
    test_jobs = df_shuffled.iloc[n_train:].to_dict(orient="records")
    return train_jobs, test_jobs
