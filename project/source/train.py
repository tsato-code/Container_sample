from config import config
from logger import logger
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import time


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
    plt.savefig(config["LGBM"]["IMPORTANCE_PATH"], bbox_inches="tight")

    # save
    with open(config["LGBM"]["MODEL_PATH"], "wb") as f:
        pickle.dump(model, f)
    logger.info(f"save {config['LGBM']['MODEL_PATH']}")


def plot_loss(result_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    result_df[["train_mape", "valid_mape"]].plot(ax=ax)
    ax.set_xscale('symlog')
    ax.set_ylabel("MAPE [%]")
    ax.set_xlabel("# iteration")
    ax.grid()
    plt.savefig(config["LGBM"]["LOSS_PATH"])


def main():
    # load feature
    with open(config["PATH"]["X_TRAIN_PATH"], "rb") as f:
        X_train = pickle.load(f)
    with open(config["PATH"]["Y_TRAIN_PATH"], "rb") as f:
        y_train = pickle.load(f)
    logger.info(f"load {config['PATH']['X_TRAIN_PATH']}")
    logger.info(f"load {config['PATH']['Y_TRAIN_PATH']}")

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
    start = time.time()
    main()
    elapsed = time.time() - start
    logger.info(f"elapsed: {elapsed:4f} [sec]")
