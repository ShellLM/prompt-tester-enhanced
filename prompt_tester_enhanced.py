#!/usr/bin/env python3
"""
Enhanced Prompt Tester - Advanced framework for testing prompt variations.
"""

import argparse
import json
import time
import sys
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import concurrent.futures
import subprocess
import os
import re
from enum import Enum

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

class SimilarityMethod(Enum):
    EXACT = "exact"
    CONTAINS = "contains"
    LEVENSHTEIN = "levenshtein"
    EMBEDDING = "embedding"
    SEMANTIC = "semantic"

@dataclass
class TestResult:
    variant_name: str
    test_case: str
    expected_output: Optional[str]
    actual_output: str
    success: bool
    latency: float
    similarity_score: Optional[float] = None
    quality_score: Optional[float] = None
    error: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class PromptVariant:
    name: str
    template: str
    description: Optional[str] = None
    version: str = "1.0"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class LLMApiClient:
    """Unified client for multiple LLM APIs."""
    
    def __init__(self, provider: str = "llm", model: str = None, api_key: str = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        
        if provider == "openai" and HAS_OPENAI:
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == "anthropic" and HAS_ANTHROPIC:
            if not api_key:
                api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key required")
            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider != "llm":
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate(self, prompt: str, test_input: str, timeout: int = 30) -> Dict[str, Any]:
        """Generate response using configured provider."""
        full_prompt = f"{prompt}\n\nInput: {test_input}\n\nOutput:"
        
        if self.provider == "llm":
            return self._call_llm_cli(full_prompt, timeout)
        elif self.provider == "openai":
            return self._call_openai(full_prompt, timeout)
        elif self.provider == "anthropic":
            return self._call_anthropic(full_prompt, timeout)
    
    def _call_llm_cli(self, prompt: str, timeout: int) -> Dict[str, Any]:
        """Call LLM CLI tool."""
        start_time = time.time()
        try:
            cmd = ['llm', '-m', self.model, '--no-log', prompt]
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
    
    def _call_openai(self, prompt: str, timeout: int) -> Dict[str, Any]:
        """Call OpenAI API."""
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model or "gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
                max_tokens=1000
            )
            
            latency = time.time() - start_time
            output = response.choices[0].message.content
            
            return {
                'success': True,
                'output': output,
                'error': None,
                'latency': latency,
                'tokens': response.usage.total_tokens if hasattr(response, 'usage') else None
            }
        except Exception as e:
            return {
                'success': False,
                'output': "",
                'error': str(e),
                'latency': time.time() - start_time
            }
    
    def _call_anthropic(self, prompt: str, timeout: int) -> Dict[str, Any]:
        """Call Anthropic API."""
        start_time = time.time()
        try:
            response = self.client.messages.create(
                model=self.model or "claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            latency = time.time() - start_time
            output = response.content[0].text
            
            return {
                'success': True,
                'output': output,
                'error': None,
                'latency': latency,
                'tokens': response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') else None
            }
        except Exception as e:
            return {
                'success': False,
                'output': "",
                'error': str(e),
                'latency': time.time() - start_time
            }

class PromptTesterEnhanced:
    """Enhanced prompt testing framework."""
    
    def __init__(self, config_path: str = None):
        self.variants = {}
        self.test_cases = []
        self.results = []
        self.llm_client = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from JSON or YAML file."""
        path = Path(config_path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML support: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        # Load variants
        if 'variants' in config:
            for name, data in config['variants'].items():
                if isinstance(data, dict):
                    self.variants[name] = PromptVariant(
                        name=name,
                        template=data['template'],
                        description=data.get('description'),
                        version=data.get('version', '1.0'),
                        tags=data.get('tags', [])
                    )
                else:
                    self.variants[name] = PromptVariant(name=name, template=data)
        
        # Load test cases
        if 'test_cases' in config:
            for tc in config['test_cases']:
                if isinstance(tc, dict):
                    self.test_cases.append({
                        'input': tc['input'],
                        'expected': tc.get('expected'),
                        'metadata': tc.get('metadata', {})
                    })
                else:
                    self.test_cases.append({'input': tc, 'expected': None, 'metadata': {}})
        
        # Load LLM settings
        if 'llm' in config:
            llm_config = config['llm']
            self.llm_client = LLMApiClient(
                provider=llm_config.get('provider', 'llm'),
                model=llm_config.get('model'),
                api_key=llm_config.get('api_key')
            )
    
    def load_variants(self, config_path: str):
        """Load prompt variants from JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'variants' in data:
            for name, template in data['variants'].items():
                self.variants[name] = PromptVariant(name=name, template=template)
    
    def load_test_cases(self, test_cases_path: str):
        """Load test cases from file with expected outputs."""
        with open(test_cases_path, 'r', encoding='utf-8') as f:
            if test_cases_path.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            self.test_cases.append(item)
                        else:
                            self.test_cases.append({'input': item, 'expected': None})
                elif isinstance(data, dict) and 'test_cases' in data:
                    for item in data['test_cases']:
                        self.test_cases.append(item)
            else:
                # Text file format: each line is "input ||| expected" or just "input"
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if '|||' in line:
                        input_part, expected_part = line.split('|||', 1)
                        self.test_cases.append({
                            'input': input_part.strip(),
                            'expected': expected_part.strip()
                        })
                    else:
                        self.test_cases.append({'input': line, 'expected': None})
    
    def calculate_similarity(self, actual: str, expected: str, 
                            method: SimilarityMethod = SimilarityMethod.EMBEDDING) -> float:
        """Calculate similarity between actual and expected output."""
        if method == SimilarityMethod.EXACT:
            return 1.0 if actual.strip() == expected.strip() else 0.0
        elif method == SimilarityMethod.CONTAINS:
            return 1.0 if expected.strip() in actual.strip() else 0.0
        elif method == SimilarityMethod.LEVENSHTEIN:
            # Simple Levenshtein implementation
            m, n = len(actual), len(expected)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    cost = 0 if actual[i-1] == expected[j-1] else 1
                    dp[i][j] = min(
                        dp[i-1][j] + 1,
                        dp[i][j-1] + 1,
                        dp[i-1][j-1] + cost
                    )
            
            max_len = max(m, n)
            return 1.0 - (dp[m][n] / max_len) if max_len > 0 else 1.0
        elif method == SimilarityMethod.EMBEDDING:
            # Placeholder for embedding-based similarity
            # Would use sentence-transformers or similar
            return 0.5  # Default
        elif method == SimilarityMethod.SEMANTIC:
            # Placeholder for semantic similarity
            return 0.5
    
    def evaluate_quality(self, output: str, expected: str = None, 
                        test_case: Dict = None) -> float:
        """Evaluate output quality on multiple dimensions."""
        score = 0.0
        weights = {
            'completeness': 0.3,
            'relevance': 0.3,
            'format': 0.2,
            'creativity': 0.2
        }
        
        # Completeness: based on length and structure
        if len(output) > 100:
            score += weights['completeness']
        
        # Relevance: keyword matching if expected exists
        if expected:
            keywords = set(re.findall(r'\b\w+\b', expected.lower()))
            output_keywords = set(re.findall(r'\b\w+\b', output.lower()))
            if keywords:
                overlap = len(keywords.intersection(output_keywords)) / len(keywords)
                score += weights['relevance'] * overlap
        else:
            score += weights['relevance'] * 0.5  # Default
        
        # Format: check for code blocks, lists, etc.
        if '```' in output or '- ' in output or '1. ' in output:
            score += weights['format']
        
        # Creativity: variety of words
        words = re.findall(r'\b\w+\b', output.lower())
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += weights['creativity'] * unique_ratio
        
        return min(score, 1.0)
    
    def run_tests(self, provider: str = None, model: str = None, 
                 parallel: int = 1, timeout: int = 30) -> List[TestResult]:
        """Run all tests."""
        if not self.llm_client:
            provider = provider or "llm"
            model = model or "gpt-3.5-turbo"
            self.llm_client = LLMApiClient(provider=provider, model=model)
        
        test_combinations = []
        for variant_name, variant in self.variants.items():
            for test_case in self.test_cases:
                test_combinations.append((variant_name, variant.template, test_case))
        
        def run_single_test(combo):
            variant_name, template, test_case = combo
            llm_result = self.llm_client.generate(template, test_case['input'], timeout)
            
            # Calculate similarity if expected output exists
            similarity_score = None
            if test_case.get('expected'):
                similarity_score = self.calculate_similarity(
                    llm_result['output'], 
                    test_case['expected'],
                    SimilarityMethod.EMBEDDING
                )
            
            # Calculate quality score
            quality_score = self.evaluate_quality(
                llm_result['output'],
                test_case.get('expected'),
                test_case
            )
            
            return TestResult(
                variant_name=variant_name,
                test_case=test_case['input'][:100],
                expected_output=test_case.get('expected'),
                actual_output=llm_result['output'],
                success=llm_result['success'],
                latency=llm_result['latency'],
                similarity_score=similarity_score,
                quality_score=quality_score,
                error=llm_result.get('error')
            )
        
        if parallel > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                self.results = list(executor.map(run_single_test, test_combinations))
        else:
            for combo in test_combinations:
                self.results.append(run_single_test(combo))
        
        return self.results
    
    def generate_report(self, output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive report."""
        if not self.results:
            raise ValueError("No test results available. Run tests first.")
        
        # Group by variant
        variant_stats = {}
        for result in self.results:
            if result.variant_name not in variant_stats:
                variant_stats[result.variant_name] = {
                    'total': 0,
                    'success': 0,
                    'total_latency': 0.0,
                    'total_similarity': 0.0,
                    'total_quality': 0.0,
                    'test_cases': []
                }
            
            stats = variant_stats[result.variant_name]
            stats['total'] += 1
            if result.success:
                stats['success'] += 1
            stats['total_latency'] += result.latency
            
            if result.similarity_score is not None:
                stats['total_similarity'] += result.similarity_score
            if result.quality_score is not None:
                stats['total_quality'] += result.quality_score
            
            stats['test_cases'].append(asdict(result))
        
        # Calculate averages
        for variant_name, stats in variant_stats.items():
            stats['success_rate'] = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            stats['avg_latency'] = stats['total_latency'] / stats['total'] if stats['total'] > 0 else 0
            
            # Calculate average similarity (only for tests with expected output)
            similarity_count = sum(1 for tc in stats['test_cases'] if tc['similarity_score'] is not None)
            if similarity_count > 0:
                stats['avg_similarity'] = stats['total_similarity'] / similarity_count
            else:
                stats['avg_similarity'] = None
            
            # Calculate average quality
            quality_count = sum(1 for tc in stats['test_cases'] if tc['quality_score'] is not None)
            if quality_count > 0:
                stats['avg_quality'] = stats['total_quality'] / quality_count
            else:
                stats['avg_quality'] = None
        
        report = {
            'metadata': {
                'timestamp': time.time(),
                'variants_tested': len(variant_stats),
                'total_tests': len(self.results),
                'llm_provider': self.llm_client.provider if self.llm_client else None,
                'llm_model': self.llm_client.model if self.llm_client else None
            },
            'summary': {
                'overall_success_rate': sum(1 for r in self.results if r.success) / len(self.results) if self.results else 0,
                'avg_latency': sum(r.latency for r in self.results) / len(self.results) if self.results else 0,
                'avg_quality': sum(r.quality_score for r in self.results if r.quality_score) / 
                              sum(1 for r in self.results if r.quality_score) if any(r.quality_score for r in self.results) else None
            },
            'variant_performance': variant_stats
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Report saved to {output_path}")
        
        return report
    
    def visualize_results(self, output_path: str = "results_visualization.png"):
        """Create visualization of test results."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not installed. Install with: pip install matplotlib")
            return
        
        if not self.results:
            print("No results to visualize")
            return
        
        # Prepare data
        variants = list(set(r.variant_name for r in self.results))
        x = range(len(variants))
        
        # Success rates
        success_rates = []
        avg_latencies = []
        avg_qualities = []
        
        for variant in variants:
            variant_results = [r for r in self.results if r.variant_name == variant]
            success_rate = sum(1 for r in variant_results if r.success) / len(variant_results)
            avg_latency = sum(r.latency for r in variant_results) / len(variant_results)
            
            quality_results = [r.quality_score for r in variant_results if r.quality_score is not None]
            avg_quality = sum(quality_results) / len(quality_results) if quality_results else 0
            
            success_rates.append(success_rate)
            avg_latencies.append(avg_latency)
            avg_qualities.append(avg_quality)
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Success Rates
        ax1.bar(x, success_rates, color='skyblue')
        ax1.set_xlabel('Variant')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate by Variant')
        ax1.set_xticks(x)
        ax1.set_xticklabels(variants, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        
        # Plot 2: Average Latency
        ax2.bar(x, avg_latencies, color='lightcoral')
        ax2.set_xlabel('Variant')
        ax2.set_ylabel('Average Latency (s)')
        ax2.set_title('Average Latency by Variant')
        ax2.set_xticks(x)
        ax2.set_xticklabels(variants, rotation=45, ha='right')
        
        # Plot 3: Average Quality
        ax3.bar(x, avg_qualities, color='lightgreen')
        ax3.set_xlabel('Variant')
        ax3.set_ylabel('Average Quality Score')
        ax3.set_title('Average Quality by Variant')
        ax3.set_xticks(x)
        ax3.set_xticklabels(variants, rotation=45, ha='right')
        ax3.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Enhanced prompt testing framework.')
    parser.add_argument('config', help='Configuration file (JSON or YAML)')
    parser.add_argument('--output', '-o', default='report.json', help='Output JSON report')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualization')
    parser.add_argument('--parallel', '-p', type=int, default=1, help='Parallel tests')
    parser.add_argument('--timeout', '-t', type=int, default=30, help='Timeout per test')
    parser.add_argument('--provider', choices=['llm', 'openai', 'anthropic'], 
                       help='LLM provider override')
    parser.add_argument('--model', '-m', help='Model override')
    
    args = parser.parse_args()
    
    try:
        tester = PromptTesterEnhanced(args.config)
        
        print(f"Testing {len(tester.variants)} variants against {len(tester.test_cases)} test cases...")
        
        results = tester.run_tests(
            provider=args.provider,
            model=args.model,
            parallel=args.parallel,
            timeout=args.timeout
        )
        
        report = tester.generate_report(args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("ENHANCED PROMPT TESTER REPORT")
        print("="*60)
        
        print(f"\nOverall Success Rate: {report['summary']['overall_success_rate']:.1%}")
        print(f"Average Latency: {report['summary']['avg_latency']:.2f}s")
        if report['summary']['avg_quality']:
            print(f"Average Quality: {report['summary']['avg_quality']:.2f}")
        
        if args.visualize:
            viz_path = args.output.replace('.json', '_viz.png')
            tester.visualize_results(viz_path)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
