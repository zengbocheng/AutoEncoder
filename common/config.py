from argparse import ArgumentParser
import yaml
import pprint
import os
import numpy as np
import torch
import random
import os.path as osp
from kogger import Logger


class Config:
    @staticmethod
    def parse_yaml(yaml_filename):
        data = Config._load_config(yaml_filename)
        Config._process_config(data)
        return Dict2Object(data)

    @staticmethod
    def get_parser():
        parser = ArgumentParser()
        parser.add_argument('--file', dest='filename', required=True)
        return parser

    @staticmethod
    def _load_config(yaml_filename):
        if os.path.exists(yaml_filename):
            with open(yaml_filename, 'r', encoding='utf-8') as stream:
                content = yaml.load(stream, Loader=yaml.FullLoader)
            return content
        else:
            raise IOError('config file at {} don\'t exist!'
                          .format(yaml_filename))

    @staticmethod
    def _set_random_seed(data):
        random.seed(data['random_seed'])
        np.random.seed(data['random_seed'])
        torch.manual_seed(data['random_seed'])
        torch.cuda.manual_seed_all(data['random_seed'])

    @staticmethod
    def _config_mkdir(data):
        for key in data:
            value = data[key]
            if type(value) is str and '/' in value:
                path = value.split('/', 1)[0]
                if len(path) > 0 and not osp.exists(path):
                    os.makedirs(path)

    @staticmethod
    def _update_path(data):
        for key in data:
            value = data[key]
            if type(value) is str:
                count = value.count('{}')
                format_values = [data['experiment_name']] + ['{}'] * count
                data[key] = value.format(*format_values)

    @staticmethod
    def _set_dtype(data):
        # set data type
        if data['precision'] == 'float64':
            data['dtype'] = torch.float64
            torch.set_default_dtype(torch.float64)
        else:
            data['dtype'] = torch.float32
            torch.set_default_dtype(torch.float32)

    @staticmethod
    def _process_config(data):
        # set random seed
        Config._set_random_seed(data)
        # set data type
        Config._set_dtype(data)
        # update path in config
        Config._update_path(data)
        # mkdir
        Config._config_mkdir(data)


class Dict2Object:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Dict2Object(value))
            else:
                setattr(self, key, value)

    def __str__(self):
        obj_str = ''
        for key, value in self.__dict__.items():
            obj_str += ' {}: {},'.format(key, str(value))
        obj_str = '{' + obj_str[1:-1] + '}'
        return obj_str


def test():
    args = Config.get_parser().parse_args()
    config = Config.parse_yaml(yaml_filename=args.filename)

    Logger.basic_config(filename=config.log_file)
    logger = Logger.get_logger(__name__)
    logger.info(config)


if __name__ == '__main__':
    test()
