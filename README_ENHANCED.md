# Enhanced Prompt Tester

A professional-grade framework for testing prompt variations with quality metrics and visualizations.

## New Features

### 1. **Multiple Quality Metrics**
- **Success Rate**: Basic pass/fail
- **Similarity Scoring**: Compare to expected outputs (Exact, Contains, Levenshtein, Embedding)
- **Quality Scoring**: Multi-dimensional evaluation (completeness, relevance, format, creativity)

### 2. **Multiple LLM Providers**
- **CLI**: Use Simon Willison's `llm` tool
- **OpenAI API**: Direct integration
- **Anthropic API**: Direct integration
- Easy to extend to other providers

### 3. **Enhanced Configuration**
- JSON and YAML support
- Version control for prompts
- Metadata and tagging
- Expected outputs for test cases

### 4. **Visualization**
- Matplotlib charts showing:
  - Success rates by variant
  - Average latency comparisons
  - Quality scores

### 5. **Better Reporting**
- Comprehensive JSON reports
- Token usage tracking (API providers)
- Timestamps and metadata

## Installation

```bash
# Basic installation
pip install -r requirements.txt

# Or run the installation script
./install_enhanced.sh
```

## Quick Start

1. Create a config file (`config.json`):
```json
{
  "variants": {
    "baseline": "Summarize:",
    "detailed": "Provide a detailed summary:"
  },
  "test_cases": [
    {
      "input": "Your text to summarize",
      "expected": "Expected summary output"
    }
  ],
  "llm": {
    "provider": "llm",
    "model": "gpt-3.5-turbo"
  }
}
```

2. Run tests:
```bash
python prompt_tester_enhanced.py config.json --output results.json --visualize
```

3. View results:
```bash
cat results.json | jq '.summary'
open results_visualization.png
```

## Advanced Usage

### Using OpenAI API directly:
```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "YOUR_API_KEY"
  }
}
```

### Custom similarity metrics:
```python
tester = PromptTesterEnhanced()
similarity = tester.calculate_similarity(
    "actual output", 
    "expected output", 
    method=SimilarityMethod.LEVENSHTEIN
)
```

## Output Format

The enhanced report includes:
```json
{
  "metadata": {
    "timestamp": 1741999200.123,
    "variants_tested": 3,
    "total_tests": 6,
    "llm_provider": "openai",
    "llm_model": "gpt-4"
  },
  "summary": {
    "overall_success_rate": 0.95,
    "avg_latency": 1.23,
    "avg_quality": 0.87
  },
  "variant_performance": {
    "baseline": {
      "success_rate": 0.9,
      "avg_latency": 0.95,
      "avg_quality": 0.82,
      "test_cases": [...]
    }
  }
}
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT
