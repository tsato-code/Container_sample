from logging import getLogger, StreamHandler, DEBUG, Formatter, FileHandler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import os
import pandas as pd
import requests
import tensorflow as tf
import zipfile


HOME_DIR = "./"
DATA_DIR = "data/raw"
DATA_PATH = os.path.join(DATA_DIR, "OnlineNewsPopularity/OnlineNewsPopularity.csv")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"


logger = getLogger(None)


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dense3 = tf.keras.layers.Dense(64, activation="relu")
        self.dense4 = tf.keras.layers.Dense(1, activation="relu")

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout1(x)
        x = self.dense3(x)
        return self.dense4(x)


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


def min_max(x):
    min_value = x.min()
    max_value = x.max()
    result = (x - min_value) / (max_value - min_value)
    return result


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
    feature = df[cols]
    feature = feature.apply(min_max)
    return feature


def plot_history_loss(axL, fit):
    axL.plot(fit.history['loss'], label="loss for training")
    axL.plot(fit.history['val_loss'], label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')


def plot_history_mape(axR, fit):
    axR.plot(fit.history['mean_absolute_percentage_error'], label="MAPE for training")
    axR.plot(fit.history['val_mean_absolute_percentage_error'], label="MAPE for validation")
    axR.set_title('model MAPE')
    axR.set_xlabel('epoch')
    axR.set_ylabel('MAPE [%]')
    axR.legend(loc='upper right')


def main():
    # get data
    get_data()
    df = pd.read_csv(DATA_PATH, header=0, index_col=None, skipinitialspace=True, engine='python')

    # split data
    train_df, test_df = train_test_split(df, test_size=9644, random_state=0)
    logger.info("# train: {}".format(train_df.shape))
    logger.info("# test: {}".format(test_df.shape))
    X = create_feature(df).values
    Y = df["shares"].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=9644, random_state=0)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=5000, random_state=0)

    # model build
    model = MLP()
    tensorboard = TensorBoard(log_dir="logs")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "./checkpoint/MLP-{epoch:04d}.ckpt",
        verbose=10,
        save_weights_only=True,
        save_freq=5000
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss="mean_absolute_percentage_error",
        metrics=["mean_absolute_percentage_error", "mean_absolute_error", "mean_squared_error"]
    )

    # model fit
    losses = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=512,
        validation_data=(x_valid, y_valid),
        callbacks=[tensorboard, checkpoint]
    )

    # evaluate
    score = model.evaluate(x_test, y_test, verbose=0)
    logger.info(score)

    # show loss
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
    plot_history_loss(axL, losses)
    plot_history_mape(axR, losses)
    plt.savefig("figure/11_nn3_loss.png")


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
