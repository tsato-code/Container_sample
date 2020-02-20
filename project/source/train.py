from logging import getLogger, StreamHandler, DEBUG, Formatter, FileHandler
from sklearn.model_selection import train_test_split
import config
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import time


logger = getLogger(None)

args = config.get_args()
yml = config.get_config(args.yaml)
yml["DIR"]["HOME_DIR"] = os.path.dirname(__file__)


def calc_mape(y_true, y_pred):
    data_num = len(y_true)
    mape = (np.sum(np.abs(y_pred - y_true) / y_true) / data_num) * 100
    return mape


def mape_func(y_pred, data):
    y_true = data.get_label()
    mape = calc_mape(y_true, y_pred)
    return "mape", mape, False


def plot_importance(model):
    feature_importance = pd.DataFrame({
        "feature_name": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain")
    })
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(111)
    sns.barplot(data=feature_importance, x="importance", y="feature_name", ax=ax)
    plt.savefig(yml["LGBM"]["IMPORTANCE_PATH"], bbox_inches="tight")

    # save
    with open(yml["LGBM"]["MODEL_PATH"], "wb") as f:
        pickle.dump(model, f)
    logger.info(f"save {yml['LGBM']['MODEL_PATH']}")


def plot_loss(result_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    result_df[["train_mape", "valid_mape"]].plot(ax=ax)
    ax.set_xscale('symlog')
    ax.set_ylabel("MAPE [%]")
    ax.set_xlabel("# iteration")
    ax.grid()
    plt.savefig(yml["LGBM"]["LOSS_PATH"])


def main():
    # load feature
    with open(yml["PATH"]["X_TRAIN_PATH"], "rb") as f:
        X_train = pickle.load(f)
    with open(yml["PATH"]["Y_TRAIN_PATH"], "rb") as f:
        y_train = pickle.load(f)
    logger.info(f"load {yml['PATH']['X_TRAIN_PATH']}")
    logger.info(f"load {yml['PATH']['Y_TRAIN_PATH']}")

    # split
    X_trn, X_val, y_trn, y_val = train_test_split(X_train, y_train, test_size=10000, random_state=0)

    # cast
    lgb_dataset_trn = lgb.Dataset(X_trn, label=y_trn, categorical_feature="auto")
    lgb_dataset_val = lgb.Dataset(X_val, label=y_val, categorical_feature="auto")

    params = {
        "objective": "rmse",
        "learning_rate": 0.1,
        "max_depth": 4
    }

    # train
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
    plot_loss(result_df)

    # plot importance
    plot_importance(model)


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
    logpath = os.path.join(yml["DIR"]["LOG_DIR"], logpath)
    handler = FileHandler(logpath, "a")
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    for key in yml:
        for param in yml[key]:
            logger.info(f"para: {param}={yml[key][param]}")

    start = time.time()
    main()
    elapsed = time.time() - start
    logger.info(f"elapsed: {elapsed:4f} [sec]")
