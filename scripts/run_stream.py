#!/usr/bin/env python3
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.streaming.server import StreamingServer
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Start Vox streaming server")
    parser.add_argument("-H", "--host", default="0.0.0.0", help="Host address")
    parser.add_argument("-p", "--port", type=int, default=8765, help="Port number")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    setup_logger(log_level="DEBUG" if args.verbose else "INFO")
    server = StreamingServer(config_path=args.config, host=args.host, port=args.port)
    print(f"Starting streaming server on ws://{args.host}:{args.port}")
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.cleanup()


if __name__ == "__main__":
    main()
