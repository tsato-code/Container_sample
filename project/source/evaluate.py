from logging import getLogger, StreamHandler, DEBUG, Formatter, FileHandler
import config
import numpy as np
import os
import pickle
import time


logger = getLogger(None)

args = config.get_args()
yml = config.get_config(args.yaml)
yml["DIR"]["HOME_DIR"] = os.path.dirname(__file__)


def calc_mape(y_true, y_pred):
    data_num = len(y_true)
    mape = (np.sum(np.abs(y_pred - y_true) / y_true) / data_num) * 100
    return mape


def main():
    # load data
    with open(yml["PATH"]["X_TEST_PATH"], "rb") as f:
        X_test = pickle.load(f)
    with open(yml["PATH"]["Y_TEST_PATH"], "rb") as f:
        y_test = pickle.load(f)
    logger.info(f"load {yml['PATH']['X_TEST_PATH']}")
    logger.info(f"load {yml['PATH']['Y_TEST_PATH']}")

    # load model
    with open(yml["LGBM"]["MODEL_PATH"], "rb") as f:
        model = pickle.load(f)

    # show test mape
    test_pred = model.predict(X_test)
    test_mape = calc_mape(y_test.values, test_pred)
    logger.info(f"test mape: {test_mape}")


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
            logger.info(f"param: {param}={yml[key][param]}")
    
    start = time.time()
    main()
    elapsed = time.time() - start
    logger.info(f"elapsed: {elapsed:4f} [sec]")
