from logger import logger
import argparse
import yaml


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", default="./project/config.yml", help="config file (.yml) path")
    parser.add_argument("-n", "--note", default=None, help="memo")
    args = parser.parse_args()
    return args


config = yaml.safe_load(open(get_args().yaml))

for key in config:
    for param in config[key]:
        logger.info(f"setting: {param}={config[key][param]}")
