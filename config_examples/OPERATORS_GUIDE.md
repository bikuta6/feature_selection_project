# Crossover and Mutation Operators Guide

This guide explains all available crossover and mutation operators for both feature synthesis and feature selection in the Feature Enhancement system.

## Feature Synthesis Operators

Feature synthesis uses Genetic Programming (GP) to evolve mathematical expressions that create new features from existing ones.

### Crossover Operators for Feature Synthesis

#### 1. Subtree Crossover (`"subtree"`) - **Default**
- **Description**: Standard GP crossover that exchanges random subtrees between two parent trees
- **How it works**: Selects random nodes in each parent and swaps their subtrees
- **Best for**: Balanced exploration and exploitation, works well in most scenarios
- **Parameters**: Uses `max_depth` from synthesis config
- **Example**: If parent1 has `(x1 + x2)` and parent2 has `(x3 * 2)`, might create `(x1 + (x3 * 2))`

#### 2. Random Crossover (`"random"`)
- **Description**: Creates new random subtrees at crossover points instead of exchanging existing ones
- **How it works**: Selects crossover points and generates completely new random subtrees
- **Best for**: High exploration, introducing novel combinations
- **Parameters**: Uses `max_depth` from synthesis config
- **Example**: Might replace existing subtrees with entirely new expressions like `(x4 / x5)`

#### 3. Point Crossover (`"point"`)
- **Description**: Exchanges nodes at corresponding positions between trees
- **How it works**: Swaps individual nodes (functions or terminals) at matching positions
- **Best for**: Fine-tuned local search, preserving overall tree structure
- **Parameters**: None specific
- **Example**: Changes `(x1 + x2)` and `(x3 - x4)` to `(x3 + x2)` and `(x1 - x4)`

### Mutation Operators for Feature Synthesis

#### 1. Subtree Mutation (`"subtree"`) - **Default**
- **Description**: Replaces a random subtree with a new random subtree
- **How it works**: Selects a random node and replaces its entire subtree
- **Best for**: Balanced diversity introduction, moderate disruption
- **Parameters**: `mutation_prob`, `max_depth`
- **Example**: `(x1 + (x2 * x3))` might become `(x1 + sin(x4))`

#### 2. Node Mutation (`"node"`)
- **Description**: Changes the value of a single node without affecting structure
- **How it works**: Replaces function names or terminal values while keeping tree structure
- **Best for**: Fine-tuning existing expressions, preserving complexity
- **Parameters**: `mutation_prob`
- **Example**: `(x1 + x2)` might become `(x1 * x2)` or `(x5 + x2)`

#### 3. Random Mutation (`"random"`)
- **Description**: Randomly selects between different mutation types
- **How it works**: Chooses subtree, node, constant, or function mutation randomly
- **Best for**: Diverse exploration strategies, adaptive mutation
- **Parameters**: `mutation_prob`, `max_depth`
- **Example**: Might perform any of the other mutation types randomly

#### 4. Parameter Mutation (`"parameter"`)
- **Description**: Only mutates terminal nodes (features and constants)
- **How it works**: Changes feature indices or adjusts constant values
- **Best for**: Fine-tuning feature usage and constant values
- **Parameters**: `mutation_prob`
- **Example**: `(x1 + 2.5)` might become `(x3 + 1.8)`

#### 5. Grow Mutation (`"grow"`)
- **Description**: Extends terminal nodes into more complex subtrees
- **How it works**: Converts simple terminals into function nodes with children
- **Best for**: Increasing expression complexity, building more sophisticated features
- **Parameters**: `mutation_prob`, `max_depth`
- **Example**: `x1` might become `(x1 + sin(x2))`

## Feature Selection Operators

Feature selection uses binary genetic algorithms where each gene represents whether a feature is selected (1) or not (0).

### Crossover Operators for Feature Selection

#### 1. Single Point Crossover (`"single_point"`) - **Default**
- **Description**: Crosses over at a single randomly selected point
- **How it works**: Splits chromosomes at one point and exchanges tails
- **Best for**: Simple, effective crossover with low disruption
- **Parameters**: None specific
- **Example**: `[1,0,1,1,0]` + `[0,1,0,1,1]` → `[1,0,0,1,1]` + `[0,1,1,1,0]`

#### 2. Two Point Crossover (`"two_point"`)
- **Description**: Exchanges the middle segment between two crossover points
- **How it works**: Selects two points and swaps the segment between them
- **Best for**: Preserving feature groups at beginning and end
- **Parameters**: None specific
- **Example**: `[1,0,1,1,0]` + `[0,1,0,1,1]` → `[1,0,0,1,0]` + `[0,1,1,1,1]`

#### 3. Uniform Crossover (`"uniform"`)
- **Description**: Each gene is independently swapped with given probability
- **How it works**: For each position, randomly decide whether to swap based on probability
- **Best for**: High mixing, exploring diverse feature combinations
- **Parameters**: `uniform_swap_prob` (default: 0.5)
- **Example**: Each bit position has 50% chance of being swapped

#### 4. Arithmetic Crossover (`"arithmetic"`)
- **Description**: Arithmetic combination of parent chromosomes converted back to binary
- **How it works**: Weighted average of parents, then threshold at 0.5
- **Best for**: Exploring intermediate solutions, smooth transitions
- **Parameters**: `arithmetic_alpha` (default: 0.5)
- **Example**: Blends feature selection patterns arithmetically

### Mutation Operators for Feature Selection

#### 1. Random Bit Flip (`"random_bit_flip"`) - **Default**
- **Description**: Each bit is independently flipped with mutation probability
- **How it works**: Goes through each gene and flips it with given probability
- **Best for**: Fine-grained exploration, standard GA mutation
- **Parameters**: `mutation_prob`
- **Example**: `[1,0,1,0,1]` might become `[0,0,1,1,1]`

#### 2. Uniform Mutation (`"uniform"`)
- **Description**: Flips exactly one randomly selected bit with given probability
- **How it works**: Selects one random position and flips it
- **Best for**: Controlled changes, minimal disruption
- **Parameters**: `mutation_prob`
- **Example**: Changes exactly one feature selection in the chromosome

#### 3. Block Mutation (`"block"`)
- **Description**: Flips a contiguous block of bits
- **How it works**: Selects a random block and flips all bits within it
- **Best for**: Exploring correlated feature groups, regional changes
- **Parameters**: `mutation_prob`, `block_size` (default: 3)
- **Example**: Flips 3 consecutive features together

#### 4. Adaptive Mutation (`"adaptive"`)
- **Description**: Adjusts mutation rate based on current feature selection ratio
- **How it works**: Higher rates when too many or too few features selected
- **Best for**: Self-regulating selection pressure, maintaining balance
- **Parameters**: `mutation_prob`, `adaptive_min_prob` (default: 0.01), `adaptive_max_prob` (default: 0.1)
- **Example**: Increases mutation when >70% or <30% features selected

## Configuration Examples

### Synthesis Configuration with Custom Operators
```json
{
  "population_size": 100,
  "max_generations": 50,
  "crossover_prob": 0.8,
  "mutation_prob": 0.1,
  "crossover_type": "random",
  "mutation_type": "parameter",
  "max_depth": 6
}
```

### Selection Configuration with Custom Operators
```json
{
  "population_size": 100,
  "generations": 50,
  "crossover_prob": 0.9,
  "mutation_prob": 0.01,
  "crossover_type": "uniform",
  "mutation_type": "adaptive",
  "uniform_swap_prob": 0.3,
  "adaptive_min_prob": 0.005,
  "adaptive_max_prob": 0.05
}
```

## Recommendations by Problem Type

### For High-Dimensional Data (Many Features)
- **Selection**: Use `"uniform"` crossover with `"adaptive"` mutation
- **Synthesis**: Use `"random"` crossover with `"grow"` mutation

### For Small Datasets
- **Selection**: Use `"single_point"` crossover with `"uniform"` mutation
- **Synthesis**: Use `"subtree"` crossover with `"parameter"` mutation

### For Complex Feature Relationships
- **Selection**: Use `"arithmetic"` crossover with `"block"` mutation
- **Synthesis**: Use `"subtree"` crossover with `"random"` mutation

### For Quick Exploration
- **Selection**: Use `"two_point"` crossover with `"random_bit_flip"` mutation
- **Synthesis**: Use `"point"` crossover with `"node"` mutation

## Usage in Main Script

You can specify operators either through configuration files or directly in the main script:

```bash
# Using configuration files
python main.py dataset.csv --synthesis-config synthesis_config.json --selection-config selection_config.json

# The configuration files should include crossover_type and mutation_type parameters
```

## Implementation Notes

1. **Default Behavior**: If no operator type is specified, the system uses the default operators
2. **Parameter Validation**: Invalid operator names will raise ValueError with available options
3. **Automatic Constraints**: All operators automatically ensure at least one feature remains selected
4. **Cross-Validation**: All fitness evaluations use cross-validation for robust performance estimation
5. **Multiprocessing**: Operator choice doesn't affect multiprocessing capabilities