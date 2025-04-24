#!/usr/bin/env python
"""
Script to run integration tests for the Financial Health Assistant
"""
import os
import sys
import subprocess
import argparse
import time
from datetime import datetime

def run_tests(test_type, verbose=False):
    """
    Run the specified tests
    
    Args:
        test_type: Type of tests to run ('integration', 'performance', or 'all')
        verbose: Whether to show detailed output
    
    Returns:
        True if all tests passed, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running {test_type.upper()} tests")
    print(f"{'='*80}")
    
    # Determine test directory
    if test_type == 'integration':
        test_dir = 'tests/integration'
    elif test_type == 'performance':
        test_dir = 'tests/performance'
    elif test_type == 'unit':
        test_dir = 'tests/unit'
    else:  # all
        test_dir = 'tests'
    
    # Build pytest command
    cmd = ['python', '-m', 'pytest', test_dir]
    
    if verbose:
        cmd.append('-v')
    
    # Run the tests
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run integration tests for Financial Health Assistant')
    parser.add_argument('--type', choices=['integration', 'performance', 'unit', 'all'], 
                        default='all', help='Type of tests to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--output', '-o', help='Output file for test results')
    
    args = parser.parse_args()
    
    # Record start time
    start_time = time.time()
    
    # Run the tests
    success = run_tests(args.type, args.verbose)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Format results
    result = "PASSED" if success else "FAILED"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = f"""
{'='*80}
TEST SUMMARY
{'='*80}
Date: {timestamp}
Test type: {args.type.upper()}
Result: {result}
Execution time: {execution_time:.2f} seconds
{'='*80}
"""
    
    print(summary)
    
    # Write to output file if specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(summary)
        print(f"Test results saved to {args.output}")
    
    # Return exit code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())