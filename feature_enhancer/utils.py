"""
Utilidades para el script main de selección de características

Este módulo contiene funciones auxiliares para configurar modelos,
operadores, datasets y parsear parámetros.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.datasets import make_regression, make_classification, load_breast_cancer, load_diabetes

from feature_enhancer.feature_selection.crossover import SinglePointCrossover, TwoPointCrossover, UniformCrossover, ArithmeticCrossover
from feature_enhancer.feature_selection.mutation import RandomBitFlip, UniformMutation, BlockMutation, AdaptiveMutation


def create_model(model_type, task_type, random_state=42):
    """
    Crea un modelo según el tipo especificado
    
    Args:
        model_type: Tipo de modelo ('linear', 'rf', 'svm')
        task_type: Tipo de tarea ('regression', 'classification')
        random_state: Semilla aleatoria
    
    Returns:
        Modelo configurado
    """
    models = {
        'regression': {
            'linear': LinearRegression(),
            'rf': RandomForestRegressor(random_state=random_state, n_estimators=50),
            'svm': SVR(kernel='rbf')
        },
        'classification': {
            'linear': LogisticRegression(random_state=random_state, max_iter=1000),
            'rf': RandomForestClassifier(random_state=random_state, n_estimators=50),
            'svm': SVC(kernel='rbf', probability=True, random_state=random_state)
        }
    }
    
    if task_type not in models:
        raise ValueError(f"Tipo de tarea no soportado: {task_type}. Use 'regression' o 'classification'")
    
    if model_type not in models[task_type]:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}. Use {list(models[task_type].keys())}")
    
    return models[task_type][model_type]


def create_crossover_operator(crossover_type, **kwargs):
    """
    Crea un operador de cruzamiento según el tipo especificado
    
    Args:
        crossover_type: Tipo de cruzamiento ('single', 'two', 'uniform', 'arithmetic')
        **kwargs: Parámetros adicionales específicos del operador
    
    Returns:
        Operador de cruzamiento
    """
    operators = {
        'single': SinglePointCrossover(),
        'two': TwoPointCrossover(),
        'uniform': UniformCrossover(swap_probability=kwargs.get('swap_prob', 0.5)),
        'arithmetic': ArithmeticCrossover(alpha=kwargs.get('alpha', 0.5))
    }
    
    if crossover_type not in operators:
        raise ValueError(f"Tipo de cruzamiento no soportado: {crossover_type}. Use {list(operators.keys())}")
    
    return operators[crossover_type]


def create_mutation_operator(mutation_type, mutation_prob, **kwargs):
    """
    Crea un operador de mutación según el tipo especificado
    
    Args:
        mutation_type: Tipo de mutación ('random', 'uniform', 'block', 'adaptive')
        mutation_prob: Probabilidad de mutación
        **kwargs: Parámetros adicionales específicos del operador
    
    Returns:
        Operador de mutación
    """
    operators = {
        'random': RandomBitFlip(mutation_prob),
        'uniform': UniformMutation(mutation_prob),
        'block': BlockMutation(mutation_prob, block_size=kwargs.get('block_size', 3)),
        'adaptive': AdaptiveMutation(
            mutation_prob, 
            min_prob=kwargs.get('min_prob', 0.01),
            max_prob=kwargs.get('max_prob', 0.1)
        )
    }
    
    if mutation_type not in operators:
        raise ValueError(f"Tipo de mutación no soportado: {mutation_type}. Use {list(operators.keys())}")
    
    return operators[mutation_type]


def load_dataset(dataset_name, **kwargs):
    """
    Carga un dataset según el nombre especificado
    
    Args:
        dataset_name: Nombre del dataset ('regression_synthetic', 'classification_synthetic', 
                     'breast_cancer', 'diabetes', 'csv')
        **kwargs: Parámetros adicionales para datasets sintéticos o ruta para CSV
    
    Returns:
        tuple: (X, y, task_type) donde task_type es 'regression' o 'classification'
    """
    if dataset_name == 'regression_synthetic':
        X, y = make_regression(
            n_samples=kwargs.get('n_samples', 1000),
            n_features=kwargs.get('n_features', 20),
            n_informative=kwargs.get('n_informative', 10),
            noise=kwargs.get('noise', 0.1),
            random_state=kwargs.get('random_state', 42)
        )
        return X, y, 'regression'
    
    elif dataset_name == 'classification_synthetic':
        X, y = make_classification(
            n_samples=kwargs.get('n_samples', 1000),
            n_features=kwargs.get('n_features', 20),
            n_informative=kwargs.get('n_informative', 10),
            n_redundant=kwargs.get('n_redundant', 5),
            n_clusters_per_class=kwargs.get('n_clusters_per_class', 1),
            random_state=kwargs.get('random_state', 42)
        )
        return X, y, 'classification'
    
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        return data.data, data.target, 'classification'
    
    elif dataset_name == 'diabetes':
        data = load_diabetes()
        return data.data, data.target, 'regression'
    
    elif dataset_name == 'csv':
        csv_path = kwargs.get('csv_path')
        target_column = kwargs.get('target_column', -1)
        
        if not csv_path:
            raise ValueError("Para dataset 'csv' debe especificar 'csv_path'")
        
        df = pd.read_csv(csv_path)
        
        if isinstance(target_column, str):
            y = df[target_column].values
            X = df.drop(columns=[target_column]).values
        else:
            y = df.iloc[:, target_column].values
            X = df.iloc[:, :target_column].values if target_column == -1 else np.concatenate([
                df.iloc[:, :target_column].values, 
                df.iloc[:, target_column+1:].values
            ], axis=1)
        
        # Determinar tipo de tarea basado en el target
        if len(np.unique(y)) <= 10:  # Heurística simple
            task_type = 'classification'
        else:
            task_type = 'regression'
        
        return X, y, task_type
    
    else:
        raise ValueError(f"Dataset no soportado: {dataset_name}")


def parse_objective_weights(weights_str):
    """
    Parsea los pesos de objetivos desde string
    
    Args:
        weights_str: String con pesos separados por coma, ej: "0.7,0.3"
    
    Returns:
        list: Lista de pesos como floats
    """
    if weights_str is None:
        return None
    
    try:
        weights = [float(w.strip()) for w in weights_str.split(',')]
        if len(weights) != 2:
            raise ValueError("Deben especificarse exactamente 2 pesos")
        if not np.isclose(sum(weights), 1.0):
            print(f"Advertencia: Los pesos no suman 1.0 (suma={sum(weights)})")
        return weights
    except Exception as e:
        raise ValueError(f"Error parseando pesos: {e}")


def print_configuration(args):
    """
    Imprime la configuración actual del experimento
    
    Args:
        args: Argumentos parseados de argparse
    """
    print("=== Configuración ===")
    print(f"Dataset: {args.dataset}")
    print(f"Modelo: {args.model}")
    print(f"Objetivo secundario: {args.secondary_objective}")
    print(f"Población: {args.population_size}, Generaciones: {args.generations}")
    print(f"Cruzamiento: {args.crossover_type} (prob={args.crossover_prob})")
    print(f"Mutación: {args.mutation_type} (prob={args.mutation_prob})")
    print(f"Pesos objetivos: {args.objective_weights}")
    print()


def print_results(selector, args):
    """
    Imprime los resultados de la selección de características
    
    Args:
        selector: FeatureSelector ajustado
        args: Argumentos parseados de argparse
    """
    print("\n=== RESULTADOS ===")
    info = selector.get_feature_importance()
    
    print(f"Características originales: {info['n_features_original']}")
    print(f"Características seleccionadas: {info['n_features_selected']}")
    print(f"Reducción de dimensionalidad: {info['reduction_ratio']*100:.1f}%")
    print(f"Características seleccionadas: {info['selected_features']}")
    
    # Determinar métrica
    metric = 'accuracy' if args.metric == 'auto' and info.get('secondary_objective_type') == 'classification' else args.metric
    if args.metric == 'auto':
        metric = 'mse'  # Por defecto para regresión
    
    print(f"\nObjetivos finales:")
    print(f"  Error (1 - ({metric} / baseline mse)): {info['objectives']['error']:.4f}")
    print(f"  {args.secondary_objective.title()}: {info['objectives'][args.secondary_objective]:.4f}")
    print(f"  Error baseline: {info['baseline_error']:.4f}")
    print(f"  Error final: {info['final_error']:.4f}")

    # Información del frente de Pareto
    pareto_front = selector.get_pareto_front()
    print(f"\nFrente de Pareto: {len(pareto_front)} soluciones")
    
    if args.verbose and len(pareto_front) > 1:
        print("\nTodas las soluciones del frente de Pareto:")
        for i, ind in enumerate(pareto_front):
            n_features = np.sum(ind.chromosome)
            print(f"  Solución {i+1}: {n_features} características, "
                 f"Error={ind.objectives[0]:.4f}, "
                 f"{args.secondary_objective}={ind.objectives[1]:.4f}")
    
    # Mostrar gráfico si se solicita
    if args.plot:
        try:
            selector.plot_pareto_front()
        except ImportError:
            print("\nAdvertencia: matplotlib no disponible, no se puede mostrar el gráfico")


def get_dataset_kwargs(args):
    """
    Prepara los kwargs para cargar el dataset
    
    Args:
        args: Argumentos parseados de argparse
    
    Returns:
        dict: Diccionario con parámetros para load_dataset
    """
    return {
        'n_samples': args.n_samples,
        'n_features': args.n_features,
        'n_informative': args.n_informative,
        'noise': args.noise,
        'random_state': args.random_state,
        'csv_path': args.csv_path,
        'target_column': args.target_column
    }


def get_crossover_kwargs(args):
    """
    Prepara los kwargs para crear el operador de cruzamiento
    
    Args:
        args: Argumentos parseados de argparse
    
    Returns:
        dict: Diccionario con parámetros para create_crossover_operator
    """
    return {
        'swap_prob': args.swap_prob,
        'alpha': args.alpha
    }


def get_mutation_kwargs(args):
    """
    Prepara los kwargs para crear el operador de mutación
    
    Args:
        args: Argumentos parseados de argparse
    
    Returns:
        dict: Diccionario con parámetros para create_mutation_operator
    """
    return {
        'block_size': args.block_size,
        'min_prob': args.min_prob,
        'max_prob': args.max_prob
    }