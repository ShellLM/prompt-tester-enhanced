#!/usr/bin/env python3
"""
Prompt Tester - A tool for testing multiple prompt variations against test cases.
"""

import argparse
import json
import time
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import concurrent.futures
import subprocess
import os

@dataclass
class TestResult:
    variant_name: str
    test_case: str
    success: bool
    latency: float
    output: str
    error: Optional[str] = None

def run_llm_prompt(model: str, prompt: str, test_input: str, timeout: int = 30) -> Dict[str, Any]:
    """Run a single LLM prompt test using llm CLI."""
    start_time = time.time()
    try:
        # Construct the full prompt
        full_prompt = f"{prompt}\n\nInput: {test_input}\n\nOutput:"
        
        # Use llm command line tool
        cmd = ['llm', '-m', model, '--no-log', full_prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        latency = time.time() - start_time
        return {
            'success': result.returncode == 0,
            'output': result.stdout.strip() if result.stdout else "",
            'error': result.stderr.strip() if result.stderr else None,
            'latency': latency
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output': "",
            'error': f"Timeout after {timeout} seconds",
            'latency': timeout
        }
    except Exception as e:
        return {
            'success': False,
            'output': "",
            'error': str(e),
            'latency': time.time() - start_time
        }

def load_variants(config_path: str) -> Dict[str, str]:
    """Load prompt variants from JSON config."""
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'variants' not in data:
        raise ValueError("Config must contain 'variants' key")
    
    return data['variants']

def load_test_cases(test_cases_path: str) -> List[str]:
    """Load test cases from file."""
    with open(test_cases_path, 'r', encoding='utf-8') as f:
        if test_cases_path.endswith('.json'):
            return json.load(f)
        else:
            # Assume text file with one test case per line
            return [line.strip() for line in f if line.strip()]

def run_test_suite(variants: Dict[str, str], test_cases: List[str], model: str = "gpt-3.5-turbo", 
                   parallel: int = 1, timeout: int = 30) -> List[TestResult]:
    """Run all variants against all test cases."""
    results = []
    
    # Generate all test combinations
    test_combinations = []
    for variant_name, prompt in variants.items():
        for test_case in test_cases:
            test_combinations.append((variant_name, prompt, test_case))
    
    def run_single_test(combo):
        variant_name, prompt, test_case = combo
        llm_result = run_llm_prompt(model, prompt, test_case, timeout)
        return TestResult(
            variant_name=variant_name,
            test_case=test_case,
            success=llm_result['success'],
            latency=llm_result['latency'],
            output=llm_result['output'],
            error=llm_result.get('error')
        )
    
    # Run tests in parallel if requested
    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            results = list(executor.map(run_single_test, test_combinations))
    else:
        for combo in test_combinations:
            results.append(run_single_test(combo))
    
    return results

def generate_report(results: List[TestResult], output_path: str = None) -> Dict[str, Any]:
    """Generate a summary report from test results."""
    
    # Group results by variant
    variant_stats = {}
    for result in results:
        if result.variant_name not in variant_stats:
            variant_stats[result.variant_name] = {
                'total': 0,
                'success': 0,
                'total_latency': 0.0,
                'test_cases': []
            }
        
        stats = variant_stats[result.variant_name]
        stats['total'] += 1
        if result.success:
            stats['success'] += 1
        stats['total_latency'] += result.latency
        stats['test_cases'].append({
            'test_case': result.test_case,
            'success': result.success,
            'latency': result.latency,
            'output_preview': result.output[:200] + '...' if len(result.output) > 200 else result.output,
            'error': result.error
        })
    
    # Calculate averages
    for variant_name, stats in variant_stats.items():
        stats['success_rate'] = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        stats['avg_latency'] = stats['total_latency'] / stats['total'] if stats['total'] > 0 else 0
    
    report = {
        'summary': {
            'total_tests': len(results),
            'variants_tested': len(variant_stats),
            'overall_success_rate': sum(1 for r in results if r.success) / len(results) if results else 0
        },
        'variant_performance': variant_stats
    }
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {output_path}")
    
    return report

def print_summary(report: Dict[str, Any]) -> None:
    """Print a human-readable summary."""
    print("\n" + "="*60)
    print("PROMPT TESTER REPORT")
    print("="*60)
    
    summary = report['summary']
    print(f"\nOverall:")
    print(f"  Total tests: {summary['total_tests']}")
    print(f"  Variants tested: {summary['variants_tested']}")
    print(f"  Overall success rate: {summary['overall_success_rate']:.1%}")
    
    print(f"\nVariant Performance:")
    print("-"*60)
    
    for variant_name, stats in report['variant_performance'].items():
        print(f"\n{variant_name}:")
        print(f"  Success rate: {stats['success_rate']:.1%} ({stats['success']}/{stats['total']})")
        print(f"  Average latency: {stats['avg_latency']:.2f}s")
        
        # Show first few test results
        print(f"  Sample results:")
        for i, test in enumerate(stats['test_cases'][:3]):
            status = "✓" if test['success'] else "✗"
            print(f"    {status} {test['test_case'][:50]}... ({test['latency']:.2f}s)")
        if len(stats['test_cases']) > 3:
            print(f"    ... and {len(stats['test_cases']) - 3} more")

def main():
    parser = argparse.ArgumentParser(description='Test prompt variations against test cases.')
    parser.add_argument('variants', help='JSON file containing prompt variants')
    parser.add_argument('test_cases', help='File containing test cases (JSON array or text file)')
    parser.add_argument('--model', '-m', default='gpt-3.5-turbo', help='LLM model to use')
    parser.add_argument('--output', '-o', help='Output JSON report file')
    parser.add_argument('--parallel', '-p', type=int, default=1, help='Number of parallel tests')
    parser.add_argument('--timeout', '-t', type=int, default=30, help='Timeout per test in seconds')
    
    args = parser.parse_args()
    
    try:
        # Load data
        print("Loading variants and test cases...")
        variants = load_variants(args.variants)
        test_cases = load_test_cases(args.test_cases)
        
        print(f"Testing {len(variants)} variants against {len(test_cases)} test cases...")
        print(f"Model: {args.model}")
        
        # Run tests
        results = run_test_suite(variants, test_cases, args.model, args.parallel, args.timeout)
        
        # Generate report
        report = generate_report(results, args.output)
        print_summary(report)
        
        # Return exit code based on overall success
        if report['summary']['overall_success_rate'] < 0.5:
            print("\nWarning: Overall success rate below 50%")
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
