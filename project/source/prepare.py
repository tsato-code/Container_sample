from config import config
from logger import logger
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
import pickle
import requests
import time
import zipfile


def get_data():
    is_exit_data = os.path.exists(os.path.join(config["DIR"]["DATA_DIR"], "OnlineNewsPopularity.zip"))
    if not is_exit_data:
        logger.info("Get dataset ...")
        try:
            res = requests.get(config["URL"]["DATA_URL"])
            logger.info(res.headers["Content-Type"])
            filename = os.path.basename(config["URL"]["DATA_URL"])
            logger.info(filename)

            with open(os.path.join(config["DIR"]["DATA_DIR"], filename), "wb") as f:
                f.write(res.content)
            logger.info("Downloaded.")

            with zipfile.ZipFile(os.path.join(config["DIR"]["DATA_DIR"], filename)) as f:
                f.extractall(config["DIR"]["DATA_DIR"])
            logger.info("Unziped.")
        except Exception as e:
            logger.info(f"Failure: {e}")
    else:
        logger.info("Dataset exists.")


def create_feature(df):
    cols = [
        'timedelta', 'n_tokens_title', 'n_tokens_content',
        'n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens',
        'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos',
        'average_token_length', 'num_keywords', 'data_channel_is_lifestyle',
        'data_channel_is_entertainment', 'data_channel_is_bus',
        'data_channel_is_socmed', 'data_channel_is_tech',
        'data_channel_is_world', 'kw_min_min', 'kw_max_min', 'kw_avg_min',
        'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg',
        'kw_avg_avg', 'self_reference_min_shares', 'self_reference_max_shares',
        'self_reference_avg_sharess', 'weekday_is_monday', 'weekday_is_tuesday',
        'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
        'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend', 'LDA_00',
        'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity',
        'global_sentiment_polarity', 'global_rate_positive_words',
        'global_rate_negative_words', 'rate_positive_words',
        'rate_negative_words', 'avg_positive_polarity', 'min_positive_polarity',
        'max_positive_polarity', 'avg_negative_polarity',
        'min_negative_polarity', 'max_negative_polarity', 'title_subjectivity',
        'title_sentiment_polarity', 'abs_title_subjectivity',
        'abs_title_sentiment_polarity'
    ]
    return df[cols]


def main():
    # get data
    get_data()
    df = pd.read_csv(config["PATH"]["DATA_PATH"], header=0, index_col=None, skipinitialspace=True, engine='python')

    # split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
    logger.info(f"# train: {len(train_df)}")
    logger.info(f"# test: {len(test_df)}")

    # get feature
    X_train = create_feature(train_df)
    y_train = np.log1p(train_df["shares"])  # transform log(1+y)
    X_test = create_feature(test_df)
    y_test = np.log1p(test_df["shares"])  # transform log(1+y)

    # serialize
    with open(config["PATH"]["X_TRAIN_PATH"], "wb") as f:
        pickle.dump(X_train, f)
    with open(config["PATH"]["Y_TRAIN_PATH"], "wb") as f:
        pickle.dump(y_train, f)
    with open(config["PATH"]["X_TEST_PATH"], "wb") as f:
        pickle.dump(X_test, f)
    with open(config["PATH"]["Y_TEST_PATH"], "wb") as f:
        pickle.dump(y_test, f)
    logger.info(f"save {config['PATH']['X_TRAIN_PATH']}")
    logger.info(f"save {config['PATH']['Y_TRAIN_PATH']}")
    logger.info(f"save {config['PATH']['X_TEST_PATH']}")
    logger.info(f"save {config['PATH']['Y_TEST_PATH']}")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    logger.info(f"elapsed: {elapsed:4f} [sec]")
