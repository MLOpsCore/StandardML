import json
import argparse

from stdml.app import WorkerApplication
from stdml.runtime.parse import InputConfig


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        prog='StdML Worker',
        description='Train and Evaluate ML models',
        epilog='version 1.0.0')

    parser.add_argument('-c', '--config', type=str, help='Path to types file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as config_file:
        config: InputConfig = InputConfig(**json.load(config_file))

    application = WorkerApplication(input_config=config)
    application.run()


if __name__ == "__main__":
    main()
