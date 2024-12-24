import subprocess
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple
import argparse

def run_model(pdf_path: str, model: str) -> float:
    """Run a single model inference and return the execution time."""
    start_time = time.time()
    subprocess.run(
        ["python", "document_classifier.py", pdf_path, f"--model={model}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return time.time() - start_time

def benchmark_models(
    pdf_path: str,
    n_iterations: int = 30,
    models: List[str] = ["phi3", "llama3.2"]
) -> pd.DataFrame:
    """Run benchmarks for multiple models."""
    results = []
    
    for model in models:
        print(f"\nBenchmarking {model}...")
        for i in range(n_iterations):
            execution_time = run_model(pdf_path, model)
            results.append({
                'model': model,
                'iteration': i + 1,
                'time': execution_time
            })
            print(f"Iteration {i + 1}/{n_iterations}: {execution_time:.3f}s")
    
    return pd.DataFrame(results)

def analyze_results(df: pd.DataFrame) -> Tuple[float, float]:
    """Perform statistical analysis on the results."""
    # Calculate basic statistics
    stats_by_model = df.groupby('model')['time'].agg(['mean', 'std', 'count'])
    print("\nSummary Statistics:")
    print(stats_by_model)
    
    # Perform t-test
    model_times = [group['time'].values for name, group in df.groupby('model')]
    t_stat, p_value = stats.ttest_ind(model_times[0], model_times[1])
    
    print(f"\nt-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    return t_stat, p_value

def plot_results(df: pd.DataFrame, output_path: str = "benchmark_results.png"):
    """Create visualizations of the benchmark results."""
    plt.figure(figsize=(12, 6))
    
    # Create subplot for boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(x='model', y='time', data=df)
    plt.title('Execution Time Distribution by Model')
    plt.ylabel('Time (seconds)')
    
    # Create subplot for violin plot
    plt.subplot(1, 2, 2)
    sns.violinplot(x='model', y='time', data=df)
    plt.title('Execution Time Distribution (Violin Plot)')
    plt.ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nPlot saved as {output_path}")

def bootstrap_analysis(df: pd.DataFrame, n_bootstrap: int = 1000):
    """Perform bootstrap analysis to estimate confidence intervals."""
    model_groups = [group['time'].values for name, group in df.groupby('model')]
    
    # Calculate mean difference in original data
    original_diff = np.mean(model_groups[0]) - np.mean(model_groups[1])
    
    # Bootstrap resampling
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        bootstrap_samples = [
            np.random.choice(group, size=len(group), replace=True)
            for group in model_groups
        ]
        bootstrap_diff = np.mean(bootstrap_samples[0]) - np.mean(bootstrap_samples[1])
        bootstrap_diffs.append(bootstrap_diff)
    
    # Calculate confidence intervals
    ci_lower, ci_upper = np.percentile(bootstrap_diffs, [2.5, 97.5])
    
    print("\nBootstrap Analysis:")
    print(f"Original mean difference: {original_diff:.4f}s")
    print(f"95% Confidence Interval: [{ci_lower:.4f}s, {ci_upper:.4f}s]")
    
    return original_diff, ci_lower, ci_upper

def main():
    parser = argparse.ArgumentParser(description='Benchmark model performance')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--iterations', type=int, default=30,
                      help='Number of iterations for each model (default: 30)')
    parser.add_argument('--bootstrap', type=int, default=1000,
                      help='Number of bootstrap samples (default: 1000)')
    args = parser.parse_args()

    # Run benchmarks
    results_df = benchmark_models(args.pdf_path, args.iterations)
    
    # Analyze results
    t_stat, p_value = analyze_results(results_df)
    
    # Bootstrap analysis
    diff, ci_lower, ci_upper = bootstrap_analysis(results_df, args.bootstrap)
    
    # Create visualizations
    plot_results(results_df)
    
    # Make a decision
    faster_model = results_df.groupby('model')['time'].mean().idxmin()
    slower_model = results_df.groupby('model')['time'].mean().idxmax()
    speed_diff_percent = abs(diff) / results_df[results_df['model'] == slower_model]['time'].mean() * 100
    
    print("\nDecision:")
    print(f"{faster_model} is faster than {slower_model} by approximately {speed_diff_percent:.1f}%")
    if p_value < 0.05:
        print("This difference is statistically significant (p < 0.05)")
    else:
        print("This difference is not statistically significant (p >= 0.05)")

if __name__ == "__main__":
    main()