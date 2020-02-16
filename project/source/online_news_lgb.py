from logging import getLogger, StreamHandler, DEBUG, Formatter, FileHandler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import seaborn as sns
import zipfile


HOME_DIR = "./"
DATA_DIR = "data/raw"
DATA_PATH = os.path.join(DATA_DIR, "OnlineNewsPopularity/OnlineNewsPopularity.csv")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"


logger = getLogger(None)


def get_data():
    is_exit_data = os.path.exists(os.path.join(DATA_DIR, "OnlineNewsPopularity.zip"))
    if not is_exit_data:
        logger.info("データセットを取得します")
        try:
            res = requests.get(DATA_URL)
            logger.info(res.headers["Content-Type"])
            filename = os.path.basename(DATA_URL)
            logger.info(filename)

            with open(os.path.join(DATA_DIR, filename), "wb") as f:
                f.write(res.content)
            logger.info("データセットをダウンロードしました")

            with zipfile.ZipFile(os.path.join(DATA_DIR, filename)) as f:
                f.extractall(DATA_DIR)
            logger.info("データセットを展開しました")

        except Exception as e:
            logger.info("データセット取得失敗: {}".format(e))

    else:
        logger.info("ダウンロード済みのデータセットがあります")


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


def calc_mape(y_true, y_pred):
    data_num = len(y_true)
    mape = (np.sum(np.abs(y_pred - y_true) / y_true) / data_num) * 100
    return mape


def mape_func(y_pred, data):
    y_true = data.get_label()
    mape = calc_mape(y_true, y_pred)
    return "mape", mape, False


def main():
    # get data
    get_data()
    df = pd.read_csv(DATA_PATH, header=0, index_col=None, skipinitialspace=True, engine='python')

    # show target variable
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    df[["shares"]].plot.hist(bins=20, ax=ax1)
    ax1.grid()
    ax2 = fig.add_subplot(1, 2, 2)
    np.log1p(df[["shares"]]).plot.hist(bins=20, ax=ax2)
    ax2.grid()
    plt.savefig("figure/01_lgb_hist_shares.png")

    # split data
    train_df, test_df = train_test_split(df, test_size=9644, random_state=0)
    logger.info("# train: {}".format(train_df.shape))
    logger.info("# test: {}".format(test_df.shape))

    X_train = create_feature(train_df)
    y_train = np.log1p(train_df["shares"])  # transform log(1+y)
    X_trn, X_val, y_trn, y_val = train_test_split(X_train, y_train, test_size=10000, random_state=0)
    lgb_dataset_trn = lgb.Dataset(X_trn, label=y_trn, categorical_feature="auto")
    lgb_dataset_val = lgb.Dataset(X_val, label=y_val, categorical_feature="auto")

    params = {
        "objective": "rmse",
        "learning_rate": 0.1,
        "max_depth": 4
    }

    model = lgb.train(
        params=params,
        train_set=lgb_dataset_trn,
        valid_sets=[lgb_dataset_val],
        num_boost_round=10000,
        early_stopping_rounds=100,
        verbose_eval=100
    )

    train_pred = model.predict(X_train)
    train_mape = calc_mape(y_train.values, train_pred)
    val_pred = model.predict(X_val)
    val_mape = calc_mape(y_val.values, val_pred)
    logger.info("train mape: {}".format(train_mape))
    logger.info("valid mape: {}".format(val_mape))

    result_dic = {}
    model = lgb.train(
        params=params,
        train_set=lgb_dataset_trn,
        valid_sets=[lgb_dataset_trn, lgb_dataset_val],
        feval=mape_func,
        num_boost_round=10000,
        verbose_eval=1000,
        evals_result=result_dic
    )

    # show loss
    result_df = pd.DataFrame(result_dic["training"]).add_prefix("train_").join(pd.DataFrame(result_dic["valid_1"]).add_prefix("valid_"))
    fig, ax = plt.subplots(figsize=(10, 6))
    result_df[["train_mape", "valid_mape"]].plot(ax=ax)
    ax.set_xscale('symlog')
    ax.set_ylabel("MAPE [%]")
    ax.set_xlabel("# iteration")
    ax.grid()
    plt.savefig("figure/02_lgb_loss.png")

    # show test mape
    X_test = create_feature(test_df)
    y_test = np.log1p(test_df["shares"])
    test_pred = model.predict(X_test)
    test_mape = calc_mape(y_test.values, test_pred)
    logger.info("test mape: {}".format(test_mape))

    # show importance
    feature_importance = pd.DataFrame({
        "feature_name": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain")
    })
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    plt.figure(figsize=(6, 12))
    sns.barplot(data=feature_importance, x="importance", y="feature_name")
    plt.savefig("figure/03_lgb_feature_importance.png")


if __name__ == "__main__":
    fmt_text = (
        "%(asctime)s %(name)s %(lineno)d"
        " [%(levelname)s][%(funcName)s] %(message)s"
    )
    log_fmt = Formatter(fmt_text)

    handler = StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(log_fmt)
    logger.setLevel("INFO")
    logger.addHandler(handler)

    logpath = os.path.basename(os.path.abspath(__file__)) + ".log"
    logpath = os.path.join(os.path.join(HOME_DIR, "logs"), logpath)
    handler = FileHandler(logpath, "a")
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    main()
