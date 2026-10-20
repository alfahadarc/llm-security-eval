import argparse
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run an LLM security evaluation experiment")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    print("Loaded config:")
    print(config)


if __name__ == "__main__":
    main()
