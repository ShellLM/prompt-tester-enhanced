# Prompt Tester

A framework for testing multiple prompt variations against test cases and generating performance reports.

## Features

- **Multiple Prompt Variants**: Test different prompt strategies side-by-side
- **Parallel Execution**: Run tests concurrently for faster results
- **Performance Metrics**: Track success rates, latency, and output quality
- **JSON Reports**: Generate detailed reports for analysis
- **Flexible Input**: JSON or text file test cases

## Prerequisites

This tool uses the [llm](https://github.com/simonw/llm) command line interface. Install it first:

```bash
pip install llm
```

Configure your preferred model:

```bash
# For OpenAI
llm keys set openai

# For other providers
llm -m <model-name> "test prompt"
```

## Installation

```bash
# Clone the repository
git clone https://github.com/irthomasthomas/prompt-tester.git
cd prompt-tester

# No Python package installation needed - uses llm CLI
```

## Usage

### Basic usage:

```bash
python prompt_tester.py config.json test_cases.json -o report.json
```

### With parallel execution:

```bash
python prompt_tester.py config.json test_cases.json -p 4 -o report.json
```

### Specify model:

```bash
python prompt_tester.py config.json test_cases.json -m gpt-4 -o report.json
```

## Configuration

### Variants JSON format:

```json
{
  "variants": {
    "variant_name": "Prompt template with {input} placeholder",
    "another_variant": "Different prompt approach"
  }
}
```

### Test cases format:

**JSON array:**
```json
[
  "Test case 1",
  "Test case 2",
  "Test case 3"
]
```

**Text file (one test case per line):**
```
Test case 1
Test case 2
Test case 3
```

## Output

The tool generates a JSON report with:
- Overall success rates
- Per-variant performance statistics
- Individual test results with latency and output previews

## Example

```bash
# Create example files
echo '{"variants": {"simple": "Summarize:", "detailed": "Provide a detailed summary:"}}' > config.json
echo '["The quick brown fox...", "Machine learning is..."]' > tests.json

# Run tests
python prompt_tester.py config.json tests.json -o results.json

# View report
cat results.json | jq '.'
```

## Options

- `--model, -m`: LLM model to use (default: gpt-3.5-turbo)
- `--output, -o`: Output JSON report file
- `--parallel, -p`: Number of parallel tests (default: 1)
- `--timeout, -t`: Timeout per test in seconds (default: 30)

## License

MIT
