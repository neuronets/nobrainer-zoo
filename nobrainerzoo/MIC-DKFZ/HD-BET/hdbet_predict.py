import os
import argparse
import yaml


def main(config):
    if config['INPUT_FILENAME']:
      INPUT_FILENAME = config['INPUT_FILENAME']
      os.system('hd-bet -i $INPUT_FILENAME')
    if config['INPUT_FOLDER']:
      INPUT_FOLDER = config['INPUT_FOLDER']
      OUTPUT_FOLDER = config['OUTPUT_FOLDER']
      os.system['hd-bet -i $INPUT_FOLDER -o $OUTPUT_FOLDER']
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, help='Path to config YAML file.')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    main(config)
