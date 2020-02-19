import argparse
import yaml


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", default="./project/config.yml", help="yaml file")
    parser.add_argument("-n", "--note", default=None, help="memo")
    args = parser.parse_args()
    return args


def get_config(yaml_file):
    with open(yaml_file) as f:
        params = yaml.safe_load(f)
    return params
