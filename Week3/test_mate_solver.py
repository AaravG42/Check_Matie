#!/usr/bin/env python3

import json
import subprocess
import sys
import os
from typing import Dict, List

def compile_cpp_program() -> bool:          
    cpp_file = "check_mate_in_n.cpp"
    executable = "check_mate_in_n"
    
    if not os.path.exists(cpp_file):
        print(f"Error: {cpp_file} not found!")
        return False
    
    try:
        subprocess.run(["g++", "-std=c++17", cpp_file, "-o", executable], check=True)
        return True
    except subprocess.CalledProcessError:
        print("Error: Failed to compile C++ program!")
        return False

def run_mate_solver(fen: str, depth: int) -> str:
    try:
        result = subprocess.run(["./check_mate_in_n", fen, str(depth)], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""

def test_positions(json_file: str, depth: int) -> None:
    try:
        with open(json_file, 'r') as f:
            positions = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file} not found!")
        return
    except json.JSONDecodeError:
        print(f"Error: {json_file} is not a valid JSON file!")
        return

    total_positions = 0
    correct_positions = 0
    
    for fen, expected_solution in positions.items():
        total_positions += 1
        solution = run_mate_solver(fen, depth)
        
        if solution == expected_solution:
            correct_positions += 1
            print(f"✓ Position {total_positions}: Correct")
        else:
            print(f"✗ Position {total_positions}: Incorrect")
            print(f"  Expected: {expected_solution}")
            print(f"  Got: {solution}")
    
    print(f"\nResults: {correct_positions}/{total_positions} correct")

def main():
    if len(sys.argv) != 3:
        print("Usage: python test_mate_solver.py <json_file> <depth>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    depth = int(sys.argv[2])
    
    if not compile_cpp_program():
        sys.exit(1)
    
    test_positions(json_file, depth)

if __name__ == "__main__":
    main() 