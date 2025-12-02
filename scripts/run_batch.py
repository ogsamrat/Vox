#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.batch import BatchPipeline
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Process audio file with Vox pipeline")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    setup_logger(log_level="DEBUG" if args.verbose else "INFO")
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    output_path = args.output or f"outputs/{audio_path.stem}_output.json"
    print(f"Processing: {audio_path}")
    pipeline = BatchPipeline(config_path=args.config)
    try:
        result = pipeline.process(str(audio_path), output_path)
        print(f"\nSaved to: {output_path}")
        print(f"\nSummary: {result.get('summary', 'N/A')[:200]}...")
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
