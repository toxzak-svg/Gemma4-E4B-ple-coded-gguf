#!/usr/bin/env python3
"""
PLE-Coded GGUF Pipeline CLI Entry Point
Usage: python -m profiling.pipeline [options]
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from profiling.pipeline.pipeline import main

if __name__ == "__main__":
    main()
