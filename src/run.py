import argparse
import platform
import yaml
from pathlib import Path


def check_environment():
    import torch
    import transformers

    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU name: {props.name}")
        print(f"GPU VRAM: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("GPU name: none")
        print("GPU VRAM: none")


def main():
    parser = argparse.ArgumentParser(description="Run an LLM security evaluation experiment")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check-environment", action="store_true",
                        help="print Python/PyTorch/Transformers/CUDA/GPU info and exit")
    args = parser.parse_args()

    if args.check_environment:
        check_environment()
        return

    if args.config is None:
        parser.error("--config is required unless --check-environment is given")

    with args.config.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    print("Loaded config:")
    print(config)


if __name__ == "__main__":
    main()
