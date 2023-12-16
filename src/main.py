import configparser
import argparse
import logging
import os
import warnings
import torch
from fl import FL

def read_config(path):
    config = configparser.ConfigParser()
    config.read(path)

    return config

def main():
    # Scrub ./config directory and grab all config files
    data_configs = [os.path.abspath(os.path.join(root, name)) for root, dirs, files in os.walk("./config") for name in files]

    # Run each config  TODO: reimplement MPI support
    for path in data_configs:
        print(path)
        config = read_config(path)
        fl = FL(config)
        fl.start()

if __name__ == "__main__":
    main()