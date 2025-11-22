# Bio-Inspired Feature Engineering Toolkit

A comprehensive feature engineering framework that combines **feature synthesis** using genetic programming with **feature selection** using the NSGA-II multi-objective optimization algorithm.

## 🎯 Overview

This project provides an automated feature engineering solution that can:

- **Synthesize new features** using genetic programming with mathematical expressions
- **Select optimal feature subsets** using NSGA-II multi-objective optimization
- **Handle both regression and classification** tasks with automatic task detection
- **Process CSV datasets** with minimal configuration required
- **Scale to high-dimensional data** with multiprocessing support
- **Provide configurable operators** for crossover and mutation strategies

The system uses bio-inspired algorithms to discover meaningful feature combinations while balancing model performance against feature sparsity.

## 🧬 Algorithm Components

### Feature Synthesis (Genetic Programming)
- **Tree-based representation** for mathematical expressions
- **Function set**: Arithmetic (`+`, `-`, `*`, `/`), trigonometric (`sin`, `cos`), logarithmic (`log`), power (`pow`), and logical operators
- **Crossover operators**: Subtree, random, and point crossover
- **Mutation operators**: Subtree replacement, node mutation, parameter mutation, grow mutation
- **Configurable depth constraints** to control expression complexity

### Feature Selection (NSGA-II)
- **Multi-objective optimization** balancing accuracy vs. sparsity/correlation/variance/information gain
- **Pareto-optimal solutions** providing trade-offs between objectives
- **Population-based evolution** with dominance ranking and crowding distance
- **Cross-validation fitness evaluation** for robust performance assessment
- **Multiple crossover types**: Single-point, two-point, uniform, arithmetic
- **Adaptive mutation strategies** with configurable rates and block operations

## 🚀 Key Features

- **Dual Enhancement Pipeline**: Synthesis → Selection in integrated workflow
- **Automatic Task Detection**: Regression/classification based on target analysis
- **Multiple ML Model Support**: Linear, tree-based, neural networks, SVM, and more
- **Rich Configuration System**: JSON-based configs with extensive examples
- **Parallel Processing**: Multiprocessing support for large datasets
- **Sklearn-Compatible**: Standard transformer interface for easy integration
- **Comprehensive Evaluation**: Cross-validation, Pareto fronts, feature importance

## 📦 Installation

### Prerequisites
- Python 3.13+
- pip or uv package manager

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd feature_selection_project

# Install with pip
pip install -e .

# Or using uv (recommended)
uv sync
```

### Dependencies

Core dependencies:
- `numpy>=2.3.4` - Numerical computing and array operations
- `pandas>=2.3.3` - Data manipulation and CSV handling
- `scikit-learn>=1.7.2` - ML models, metrics, and preprocessing
- `matplotlib>=3.10.7` - Plotting and visualization
- `tqdm>=4.67.1` - Progress bars for long-running operations

## 🏃‍♂️ Quick Start

### Basic Usage

```bash
# Feature selection only (default behavior)
python main.py data/California.csv

# Specify target column by name
python main.py data/Student_Performance.csv --target "G3"

# Use different ML model
python main.py data/Wine.csv --model rf

# Enable both synthesis and selection
python main.py data/Happy.csv \
    --synthesis-config configs/synthesis_config.json \
    --selection-config configs/selection_config.json
```

### Advanced Usage

```bash
# High-performance mode with multiprocessing
python main.py data/Mnist.csv --use-multiprocessing --n-jobs -1

# Custom test split and scaling
python main.py data/Diabetes.csv --test-size 0.3 --no-scale

# Quiet mode with specific random seed
python main.py data/Wine.csv --quiet --random-state 123
```

## ⚙️ Configuration

### Feature Selection Configuration

```json
{
  "population_size": 100,
  "generations": 50,
  "secondary_objective": "sparsity",
  "metric": "accuracy",
  "crossover_type": "uniform",
  "mutation_type": "adaptive",
  "mutation_prob": 0.01,
  "uniform_swap_prob": 0.3,
  "objective_weights": [0.7, 0.3]
}
```

**Secondary Objectives:**
- `"sparsity"` - Minimize number of selected features
- `"correlation"` - Minimize feature correlation
- `"variance"` - Maximize feature variance
- `"information_gain"` - Maximize information content

### Feature Synthesis Configuration

```json
{
  "population_size": 100,
  "max_generations": 50,
  "max_depth": 6,
  "crossover_type": "subtree",
  "mutation_type": "parameter",
  "mutation_prob": 0.1,
  "tournament_size": 3
}
```

**Crossover Types:**
- `"subtree"` - Standard GP subtree exchange
- `"random"` - Creates new random subtrees
- `"point"` - Exchanges nodes at positions

**Mutation Types:**
- `"subtree"` - Replaces random subtree
- `"node"` - Changes individual nodes
- `"parameter"` - Mutates only terminals
- `"grow"` - Extends terminals into subtrees
- `"random"` - Randomly selects strategy

### Usage Scenarios

**Novel Feature Discovery (High Exploration):**
```json
{
  "crossover_type": "random",
  "mutation_type": "random",
  "mutation_prob": 0.15
}
```

**Fine-Tuning Existing Features:**
```json
{
  "crossover_type": "point",
  "mutation_type": "parameter",
  "mutation_prob": 0.05
}
```

**High-Dimensional Data:**
```json
{
  "crossover_type": "uniform",
  "mutation_type": "adaptive",
  "uniform_swap_prob": 0.3
}
```

## 🔧 Python API

### Basic Integration

```python
from feature_enhancer import FeatureEnhancer, DatasetLoader
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess data
X, y = DatasetLoader.load_csv('data/California.csv')
X, y = DatasetLoader.preprocess_dataset(X, y)

# Configure enhancement
enhancer = FeatureEnhancer(
    synthesis_config={
        "population_size": 50,
        "max_generations": 30,
        "max_depth": 4,
        "crossover_type": "subtree",
        "mutation_type": "parameter"
    },
    selection_config={
        "population_size": 100,
        "generations": 50,
        "secondary_objective": "sparsity",
        "crossover_type": "uniform",
        "mutation_type": "adaptive"
    },
    verbose=True
)

# Apply enhancement
model = RandomForestRegressor()
X_enhanced = enhancer.fit_transform(X, y, model)

# Analyze results
feature_info = enhancer.get_feature_info()
pareto_front = enhancer.get_pareto_front()
```

### Advanced API Usage

```python
# Selection only workflow
selector = FeatureSelector(
    model=model,
    secondary_objective="correlation",
    population_size=100,
    generations=50
)

X_selected = selector.fit_transform(X, y)
selector.plot_pareto_front()  # Visualize trade-offs

# Synthesis only workflow
synthesizer = MultiFeatureGA(
    population_size=100,
    max_generations=50,
    max_depth=6
)

new_features = synthesizer.evolve_multiple_features(X, y, n_features=5)
```

## 📊 Example Datasets

The `data/` directory includes diverse example datasets:

| Dataset | Type | Features | Samples | Description |
|---------|------|----------|---------|-------------|
| `California.csv` | Regression | 8 | 20,640 | California housing prices |
| `Wine.csv` | Classification | 13 | 178 | Wine quality classification |
| `Diabetes.csv` | Regression | 10 | 442 | Diabetes progression |
| `AutoMPG.csv` | Regression | 8 | 398 | Auto fuel efficiency prediction |
| `Fish.csv` | Classification | 6 | 159 | Fish species classification |
| `Happy.csv` | Regression | 11 | 156 | World happiness index |

## 🎛️ Command Line Interface

```bash
python main.py dataset.csv [OPTIONS]

Positional Arguments:
  csv_path                  Path to CSV dataset

Target Configuration:
  -t, --target TARGET       Target column name or index (default: -1)

Model Selection:
  -m, --model MODEL         Model choice: auto, linear, logistic, rf, ridge,
                           lasso, knn, svm, dt, gb, mlp (default: ridge)

Configuration Files:
  --synthesis-config FILE   Path to synthesis configuration JSON
  --selection-config FILE   Path to selection configuration JSON

Data Processing:
  --no-scale               Disable feature scaling
  --test-size FLOAT        Test set proportion (default: 0.2)

Performance Options:
  --use-multiprocessing    Enable parallel processing
  --n-jobs N               Number of processes (-1 for all cores)

Reproducibility:
  --random-state INT       Random seed (default: 42)

Output Control:
  -q, --quiet              Reduce output verbosity
```

## 📁 Project Structure

```
feature_selection_project/
├── feature_enhancer/              # Main package
│   ├── __init__.py               # Package exports
│   ├── feature_enhancer.py       # Main FeatureEnhancer class
│   ├── dataset_utils.py          # Data loading and preprocessing
│   ├── utils.py                  # Utility functions
│   ├── feature_selection/        # NSGA-II implementation
│   │   ├── __init__.py
│   │   ├── feature_selector.py   # Main selector class
│   │   ├── nsga2.py             # NSGA-II algorithm
│   │   ├── individual.py        # Individual representation
│   │   ├── fitness.py           # Fitness functions
│   │   ├── crossover.py         # Crossover operators
│   │   └── mutation.py          # Mutation operators
│   └── feature_synthesis/        # Genetic Programming
│       ├── __init__.py
│       ├── feature_synthesis.py  # GP algorithms
│       ├── individual.py        # GP tree representation
│       ├── crossover.py         # GP crossover operators
│       └── mutation.py          # GP mutation operators
├── config_examples/              # Configuration examples and guides
│   ├── README.md                # Configuration documentation
│   ├── OPERATORS_GUIDE.md       # Detailed operator descriptions
│   ├── synthesis_config_example.json
│   ├── selection_config_example.json
│   └── advanced_example.py      # Python examples
├── configs/                     # Pre-configured parameter sets
│   ├── quick/                   # Fast execution configs
│   │   ├── quick_selection_*.json    # Quick selection configs
│   │   └── quick_synthesis_*.json    # Quick synthesis configs
│   ├── medium/                  # Balanced performance configs
│   │   ├── medium_selection_*.json   # Medium selection configs
│   │   └── medium_synthesis_*.json   # Medium synthesis configs
│   └── slow/                    # High-quality, longer-running configs
│       ├── slow_selection_*.json     # Thorough selection configs
│       └── slow_synthesis_*.json     # Thorough synthesis configs
├── data/                        # Example datasets
│   ├── AutoMPG.csv             # Auto MPG regression dataset
│   ├── California.csv          # California housing prices
│   ├── Diabetes.csv            # Diabetes progression dataset
│   ├── Fish.csv                # Fish market classification
│   ├── Happy.csv               # World happiness index
│   └── Wine.csv                # Wine quality classification
├── comparison_results/          # Algorithm comparison outputs
│   ├── comparison_visualization.png  # Performance comparison plots
│   ├── latest_comparison_results.csv # Detailed results data
│   └── summary_report.txt       # Analysis summary
├── main.py                      # Command-line interface
├── comparison_analysis.py       # Algorithm comparison tool
├── run_comparison.py           # Automated comparison runner
├── visualize_results.py        # Results visualization utility
├── test_default_config.py      # Configuration testing
├── pyproject.toml              # Project configuration and dependencies
├── uv.lock                     # Dependency lock file
├── .gitignore                  # Git ignore patterns
├── .python-version             # Python version specification
└── README.md                   # This file
```

## 🔬 Algorithm Details

### NSGA-II Multi-Objective Optimization

1. **Population Initialization**: Random binary chromosomes representing feature subsets
2. **Fitness Evaluation**: Cross-validated model performance + secondary objective
3. **Non-Dominated Sorting**: Rank solutions by Pareto dominance
4. **Crowding Distance**: Maintain population diversity
5. **Selection**: Tournament selection based on rank and crowding distance
6. **Reproduction**: Apply crossover and mutation operators
7. **Environmental Selection**: Select best individuals for next generation

### Genetic Programming Tree Evolution

1. **Tree Initialization**: Random mathematical expressions within depth constraints
2. **Fitness Evaluation**: Feature usefulness via cross-validated model improvement
3. **Tournament Selection**: Select parents based on fitness
4. **Tree Crossover**: Exchange subtrees between parent expressions
5. **Tree Mutation**: Modify nodes, parameters, or subtrees
6. **Population Replacement**: Generational or steady-state strategies

### Integration Workflow

1. **Synthesis Phase**: Generate N new features using GP
2. **Combination Phase**: Merge original and synthesized features
3. **Selection Phase**: Apply NSGA-II to find optimal feature subset
4. **Evaluation Phase**: Cross-validate final feature set performance

## 🛠️ Development

### Running Tests

```bash
# Test operator functionality
python config_examples/test_operators.py

# Test with default configuration
python test_default_config.py

# Run advanced examples
python config_examples/advanced_example.py
```

### Performance Optimization

- Use `--use-multiprocessing` for large datasets
- Reduce `population_size` and `generations` for faster prototyping
- Enable `--no-scale` if features are already normalized
- Use `"sparsity"` secondary objective for aggressive feature reduction
