#!/usr/bin/env python3
"""
Clinical Digital Twin — Pipeline Entry Point

Usage
-----
    python run_pipeline.py                    # standard mode (recommended first run)
    python run_pipeline.py --full             # load entire large tables (hours)
    python run_pipeline.py --skip-large       # small tables only (fast smoke test)
    python run_pipeline.py --steps load clean # run specific steps
"""

from src.data.pipeline import main

if __name__ == "__main__":
    main()
