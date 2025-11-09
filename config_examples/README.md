# Crossover and Mutation Operator Configuration

This directory contains examples and documentation for the new crossover and mutation operator configuration functionality in the Feature Enhancement system.

## What's New

The Feature Enhancement system now supports configurable crossover and mutation operators for both feature synthesis and feature selection components. This allows you to customize the genetic algorithm behavior based on your specific problem characteristics and requirements.

## Quick Start

### Basic Usage

Add `crossover_type` and `mutation_type` parameters to your configuration files:

**Synthesis Configuration (`synthesis_config.json`):**
```json
{
  "population_size": 100,
  "max_generations": 50,
  "crossover_type": "random",
  "mutation_type": "parameter",
  "mutation_prob": 0.1,
  "max_depth": 6
}
```

**Selection Configuration (`selection_config.json`):**
```json
{
  "population_size": 100,
  "generations": 50,
  "crossover_type": "uniform",
  "mutation_type": "adaptive",
  "mutation_prob": 0.01,
  "uniform_swap_prob": 0.3
}
```

### Run with Custom Operators

```bash
python main.py dataset.csv --synthesis-config synthesis_config.json --selection-config selection_config.json
```

## Available Operators

### Feature Synthesis Operators

**Crossover Types:**
- `"subtree"` (default) - Standard GP subtree exchange
- `"random"` - Creates new random subtrees at crossover points
- `"point"` - Exchanges nodes at corresponding positions

**Mutation Types:**
- `"subtree"` (default) - Replaces random subtree with new one
- `"node"` - Changes individual node values
- `"random"` - Randomly selects mutation strategy
- `"parameter"` - Only mutates terminals (features/constants)
- `"grow"` - Extends terminals into more complex subtrees

### Feature Selection Operators

**Crossover Types:**
- `"single_point"` (default) - Single-point crossover
- `"two_point"` - Two-point crossover
- `"uniform"` - Uniform crossover with configurable swap probability
- `"arithmetic"` - Arithmetic combination of parents

**Mutation Types:**
- `"random_bit_flip"` (default) - Independent bit flipping
- `"uniform"` - Single random bit flip
- `"block"` - Contiguous block mutation
- `"adaptive"` - Adapts mutation rate based on selection ratio

## Files in This Directory

- `README.md` - This file
- `OPERATORS_GUIDE.md` - Comprehensive documentation of all operators
- `synthesis_config_example.json` - Example synthesis configuration
- `selection_config_example.json` - Example selection configuration
- `advanced_example.py` - Practical examples for different scenarios
- `test_operators.py` - Test script to verify functionality

## Usage Scenarios

### High Exploration (Novel Feature Discovery)
```json
{
  "crossover_type": "random",
  "mutation_type": "random",
  "mutation_prob": 0.15
}
```

### Fine-Tuning (Precise Optimization)
```json
{
  "crossover_type": "point",
  "mutation_type": "parameter",
  "mutation_prob": 0.05
}
```

### Correlated Features (Group Selection)
```json
{
  "crossover_type": "two_point",
  "mutation_type": "block",
  "block_size": 3
}
```

### High-Dimensional Data (Aggressive Selection)
```json
{
  "crossover_type": "uniform",
  "mutation_type": "random_bit_flip",
  "uniform_swap_prob": 0.3
}
```

## Additional Parameters

Some operators support additional configuration parameters:

- `uniform_swap_prob` - For uniform crossover (default: 0.5)
- `arithmetic_alpha` - For arithmetic crossover (default: 0.5)
- `block_size` - For block mutation (default: 3)
- `adaptive_min_prob` - For adaptive mutation (default: 0.01)
- `adaptive_max_prob` - For adaptive mutation (default: 0.1)

## Testing Your Configuration

Before running on your dataset, test your configuration:

```bash
python config_examples/test_operators.py
```

This will verify that your operator choices are valid and working correctly.

## Advanced Examples

Run the advanced examples to see different operator strategies in action:

```bash
python config_examples/advanced_example.py
```

## Default Behavior

If you don't specify operator types, the system uses:
- **Synthesis**: `subtree` crossover + `subtree` mutation
- **Selection**: `single_point` crossover + `random_bit_flip` mutation

This ensures backward compatibility with existing configurations.

## Best Practices

1. **Start Simple**: Begin with default operators and gradually experiment
2. **Match Problem Type**: Use exploratory operators for discovery, conservative for fine-tuning
3. **Consider Data Size**: Use adaptive/uniform operators for high-dimensional data
4. **Test Multiple Strategies**: Run comparisons to find optimal operators
5. **Document Results**: Keep track of which operators work best for your datasets

## Troubleshooting

**Invalid Operator Error**: Check that operator names match exactly (case-sensitive)
**Poor Performance**: Try different operator combinations or adjust probabilities
**Slow Evolution**: Consider reducing population size or generations first

For detailed operator descriptions and recommendations, see `OPERATORS_GUIDE.md`.