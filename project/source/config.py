import argparse
import yaml


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", default="./project/config.yml", help="config file (.yml) path")
    parser.add_argument("-n", "--note", default=None, help="memo")
    args = parser.parse_args()
    return args


config = yaml.safe_load(open(get_args().yaml))
