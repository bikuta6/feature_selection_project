#!/usr/bin/env python3
"""
Simple runner script for the feature enhancement comparison analysis.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the comparison analysis."""
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Path to the comparison script
    comparison_script = project_root / "comparison_analysis.py"
    
    if not comparison_script.exists():
        print(f"Error: {comparison_script} not found!")
        return 1
    
    print("Starting Feature Enhancement Comparison Analysis...")
    print("=" * 60)
    print("This will test feature enhancement on all datasets using multiple regression models.")
    print("The enhancement process will use Ridge regression as the base model.")
    print("Results will be saved to the 'comparison_results' directory.")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(project_root)
    
    # Run the comparison analysis
    try:
        # Use uv run if available, otherwise use python directly
        cmd = ["uv", "run", "python", str(comparison_script)]
        
        # Try uv run first
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            return result.returncode
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If uv is not available or fails, try with python directly
            print("uv not available, trying with python directly...")
            cmd = [sys.executable, str(comparison_script)]
            result = subprocess.run(cmd, check=True, capture_output=False)
            return result.returncode
            
    except subprocess.CalledProcessError as e:
        print(f"\nError running comparison analysis: {e}")
        return e.returncode
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())