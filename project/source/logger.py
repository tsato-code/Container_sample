from logging import getLogger, StreamHandler, DEBUG, Formatter
from logging.handlers import TimedRotatingFileHandler
import os


def get_logger():
    logger = getLogger(None)

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

    logdir, logfile = os.path.split(os.path.abspath(__file__))
    logpath = os.path.join(logdir, os.pardir, "logs", logfile + ".log")
    handler = TimedRotatingFileHandler(
        filename=logpath,
        when="D",
        backupCount=7
    )
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    return logger


logger = get_logger()
