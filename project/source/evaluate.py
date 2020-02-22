from config import config
from logger import logger
import pickle
import time
from utils import calc_mape


def main():
    # load data
    with open(config["PATH"]["X_TEST_PATH"], "rb") as f:
        X_test = pickle.load(f)
    with open(config["PATH"]["Y_TEST_PATH"], "rb") as f:
        y_test = pickle.load(f)
    logger.info(f"load {config['PATH']['X_TEST_PATH']}")
    logger.info(f"load {config['PATH']['Y_TEST_PATH']}")

    # load model
    with open(config["LGBM"]["MODEL_PATH"], "rb") as f:
        model = pickle.load(f)
    logger.info(f"load {config['LGBM']['MODEL_PATH']}")

    # show test mape
    test_pred = model.predict(X_test)
    test_mape = calc_mape(y_test.values, test_pred)
    logger.info(f"test mape: {test_mape}")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    logger.info(f"elapsed: {elapsed:4f} [sec]")
