# Feature Enhancement Project

A bio-inspired feature engineering toolkit that combines **feature synthesis** using genetic programming with **feature selection** using the NSGA-II multi-objective optimization algorithm.

## 🎯 Overview

This project provides a comprehensive solution for automated feature engineering that can:

- **Synthesize new features** using genetic programming techniques
- **Select optimal feature subsets** using NSGA-II optimization
- **Handle both regression and classification** tasks automatically
- **Process CSV datasets** with minimal configuration
- **Scale to high-dimensional data** with multiprocessing support

The system uses bio-inspired algorithms to discover meaningful feature combinations and reduce dimensionality while maintaining or improving model performance.

## 🚀 Key Features

- **Dual Enhancement**: Feature synthesis + selection in one pipeline
- **Multi-objective Optimization**: Balance accuracy vs. sparsity using NSGA-II
- **Automatic Task Detection**: Handles regression/classification automatically
- **Multiple ML Models**: Support for linear models, tree-based, neural networks, and more
- **Configurable Operators**: Customizable crossover and mutation strategies
- **Parallel Processing**: Multiprocessing support for large datasets
- **Rich Configuration**: JSON-based configuration with examples

## 📦 Installation

### Prerequisites

- Python 3.13+
- pip or uv package manager

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd feature_selection_project

# Install dependencies
pip install -e .

# Or using uv (recommended)
uv sync
```

### Dependencies

The project requires:
- `numpy>=2.3.4` - Numerical computing
- `pandas>=2.3.3` - Data manipulation
- `scikit-learn>=1.7.2` - Machine learning models and metrics
- `matplotlib>=3.10.7` - Visualization
- `tqdm>=4.67.1` - Progress bars

## 🏃‍♂️ Quick Start

### Basic Usage

```bash
# Run with default settings (feature selection only)
python main.py data/California.csv

# Specify target column
python main.py data/Student_Performance.csv --target "G3"

# Use different model
python main.py data/Wine.csv --model rf
```

### With Configuration Files

```bash
# Feature selection only
python main.py data/Diabetes.csv --selection-config configs/selection_config.json

# Feature synthesis + selection
python main.py data/Happy.csv \
    --synthesis-config configs/synthesis_config.json \
    --selection-config configs/selection_config.json

# Enable multiprocessing for large datasets
python main.py data/Mnist.csv --use-multiprocessing --n-jobs -1
```

## 📊 Example Datasets

The `data/` directory includes several example datasets:

- **California.csv** - California housing prices (regression)
- **Wine.csv** - Wine quality classification
- **Diabetes.csv** - Diabetes progression (regression)
- **Student_Performance.csv** - Student academic performance
- **Happy.csv** - World happiness index
- **Mnist.csv** - MNIST digit classification subset

## ⚙️ Configuration

### Feature Selection Configuration

Create a JSON file for NSGA-II parameters:

```json
{
    "population_size": 100,
    "generations": 50,
    "secondary_objective": "sparsity",
    "metric": "accuracy",
    "crossover_type": "uniform",
    "mutation_type": "adaptive",
    "mutation_prob": 0.01
}
```

### Feature Synthesis Configuration

Configure genetic programming for feature creation:

```json
{
    "population_size": 100,
    "max_generations": 50,
    "max_depth": 6,
    "crossover_type": "subtree",
    "mutation_type": "parameter",
    "mutation_prob": 0.1
}
```

See `config_examples/` for detailed examples and operator guides.

## 🔧 Advanced Usage

### Python API

```python
from feature_enhancer import FeatureEnhancer, DatasetLoader
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess data
X, y = DatasetLoader.load_csv('data/California.csv')
X, y = DatasetLoader.preprocess_dataset(X, y)

# Configure enhancement
synthesis_config = {
    "population_size": 50,
    "max_generations": 30,
    "max_depth": 4
}

selection_config = {
    "population_size": 100,
    "generations": 50,
    "secondary_objective": "sparsity"
}

# Create enhancer
enhancer = FeatureEnhancer(
    synthesis_config=synthesis_config,
    selection_config=selection_config,
    verbose=True
)

# Apply enhancement
model = RandomForestRegressor()
X_enhanced = enhancer.fit_transform(X, y, model)

# Get enhancement details
feature_info = enhancer.get_feature_info()
selected_features = enhancer.get_selected_features_summary()
```

### Multiprocessing Support

For large datasets or complex synthesis:

```python
enhancer = FeatureEnhancer(
    synthesis_config=config,
    use_multiprocessing=True,
    n_jobs=-1  # Use all available cores
)
```

## 🎛️ Command Line Options

```bash
python main.py dataset.csv [OPTIONS]

Positional Arguments:
  csv_path                  Path to CSV dataset

Optional Arguments:
  -t, --target TARGET       Target column name or index (default: -1)
  -m, --model MODEL         Model choice: auto, linear, logistic, rf, ridge,
                           lasso, knn, svm, dt, gb, mlp (default: ridge)
  --synthesis-config FILE   Path to synthesis config JSON
  --selection-config FILE   Path to selection config JSON
  --no-scale               Disable feature scaling
  --use-multiprocessing    Enable parallel processing
  --n-jobs N               Number of processes (-1 for all cores)
  --test-size FLOAT        Test set proportion (default: 0.2)
  --random-state INT       Random seed (default: 42)
  -q, --quiet              Reduce output verbosity
```

## 🧬 Algorithm Details

### Feature Synthesis (Genetic Programming)
- Creates new features through mathematical combinations
- Uses tree-based genetic programming
- Supports arithmetic, trigonometric, and logical operators
- Configurable depth and complexity constraints

### Feature Selection (NSGA-II)
- Multi-objective optimization balancing accuracy and sparsity
- Pareto-optimal solutions for different trade-offs
- Population-based evolutionary algorithm
- Maintains diversity through crowding distance

### Integration
1. **Synthesis Phase**: Generate candidate features using GP
2. **Combination Phase**: Merge original and synthesized features
3. **Selection Phase**: Apply NSGA-II to find optimal subset
4. **Evaluation Phase**: Cross-validate performance improvements

## 📁 Project Structure

```
feature_selection_project/
├── feature_enhancer/           # Main package
│   ├── feature_synthesis/      # Genetic programming
│   ├── feature_selection/      # NSGA-II implementation
│   ├── feature_enhancer.py     # Main interface
│   └── dataset_utils.py        # Data handling utilities
├── config_examples/            # Configuration examples
├── data/                       # Example datasets
├── configs/                    # User configurations
├── main.py                     # Command-line interface
└── README.md                   # This file
```
