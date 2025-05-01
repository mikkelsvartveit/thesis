import numpy as np
from scipy import stats


# def calculate_confidence_interval(accuracies: list[float], confidence=0.95):
#     """
#     Calculate confidence interval across all runs for model accuracies

#     Parameters:
#     accuracies (list): List of accuracy values from multiple runs
#     confidence (float): Confidence level (default: 0.95 for 95% CI)

#     Returns:
#     tuple: (mean, lower_bound, upper_bound)
#     """
#     n = len(accuracies)
#     mean = np.mean(accuracies)
#     std_dev = np.std(accuracies, ddof=1)  # Using ddof=1 for sample standard deviation

#     # Calculate confidence interval using t-distribution
#     t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
#     margin_error = t_critical * (std_dev / np.sqrt(n))

#     lower_bound = mean - margin_error
#     upper_bound = mean + margin_error

#    return mean, margin_error, lower_bound, upper_bound


def calculate_confidence_interval(accuracies, n_samples=50, confidence=0.95):
    """
    Calculate confidence interval for model accuracies considering both
    seed variability and sample size

    Parameters:
    accuracies (list): List of accuracy values from multiple runs
    n_samples (int): Number of classification samples per run
    confidence (float): Confidence level (default: 0.95 for 95% CI)

    Returns:
    tuple: (mean, lower_bound, upper_bound)
    """
    n_runs = len(accuracies)
    mean_acc = np.mean(accuracies)

    # Step 1: Calculate variance from seed-to-seed differences
    variance_between_runs = np.var(accuracies, ddof=1)

    # Step 2: Calculate variance from finite sample size (binomial variance)
    # For each accuracy value, the estimated variance is p(1-p)/n
    # where p is the accuracy and n is the number of samples
    variances_within_runs = [acc * (1 - acc) / n_samples for acc in accuracies]
    mean_variance_within = np.mean(variances_within_runs)

    # Step 3: Combine the variances
    total_variance = variance_between_runs + mean_variance_within

    # Step 4: Calculate confidence interval using t-distribution
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n_runs - 1)
    margin_error = t_critical * np.sqrt(total_variance / n_runs)

    lower_bound = mean_acc - margin_error
    upper_bound = mean_acc + margin_error

    return mean_acc, margin_error, lower_bound, upper_bound


def compare_model_accuracies(
    model_a_accuracies: list[float], model_b_accuracies: list[float], alpha=0.05
):
    """
    Perform paired t-test to compare the runs accross two models

    Parameters:
    model_a_accuracies: List of accuracy values for model A
    model_b_accuracies: List of accuracy values for model B
    alpha: Significance level (default: 0.05 for 95% confidence)

    Returns:
    tuple: (significant_difference, p_value, mean_difference)
    """
    # Ensure the accuracies are from the same seeds/runs
    assert len(model_a_accuracies) == len(model_b_accuracies)

    # Calculate differences
    differences = np.array(model_b_accuracies) - np.array(model_a_accuracies)

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(model_b_accuracies, model_a_accuracies)

    # Determine if difference is significant
    significant = p_value < alpha
    mean_diff = np.mean(differences)

    return significant, p_value, mean_diff


if __name__ == "__main__":
    # Example usage
    model_1_accuracies = [0.82, 0.85, 0.83, 0.84, 0.86]
    mean, err, lower, upper = calculate_confidence_interval(model_1_accuracies)
    print(f"Model accuracy: {mean:.4f}+-{err:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
    mean, err, lower, upper = calculate_confidence_interval2(model_1_accuracies)
    print(f"Model accuracy: {mean:.4f}+-{err:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")

    model_2_accuracies = [0.78, 0.80, 0.79, 0.81, 0.82]
    significant, p_value, mean_diff = compare_model_accuracies(
        model_1_accuracies, model_2_accuracies
    )
    print(
        f"Model comparison: Significant difference: {significant}, p-value: {p_value:.4f}, Mean difference: {mean_diff:.4f}"
    )
