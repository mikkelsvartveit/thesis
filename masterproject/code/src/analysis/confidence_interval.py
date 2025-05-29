import numpy as np
from scipy import stats


def calculate_weighted_run_average(fold_results, fold_sample_counts):
    """
    Calculate weighted average accuracy across folds in a single run. Used for LOGO-CV and K-Fold CV.

    Parameters:
    fold_accuracies (list): List of accuracy values for each fold
    fold_sample_counts (list): List of sample counts for each fold

    Returns:
    float: Weighted average accuracy
    """
    assert len(fold_results) == len(
        fold_sample_counts
    ), "Number of fold_results must match number of sample counts"

    total_samples = sum(fold_sample_counts)
    weighted_res = (
        sum(res * count for res, count in zip(fold_results, fold_sample_counts))
        / total_samples
    )

    return weighted_res


def calculate_confidence_interval(accuracies, n_samples, confidence=0.95):
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


def logo_cv_confidence_interval(
    all_run_fold_accuracies: list[list[float]],
    fold_sample_counts: list[list[int]],
    confidence=0.95,
):
    """
    Calculate confidence interval for LOGO-CV with multiple runs

    Parameters:
    all_run_fold_accuracies: List of lists, where each inner list contains fold accuracies for one run
    fold_sample_counts: List of lists, where each inner list contains sample counts for each fold in one run
    confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
    tuple: (mean, lower_bound, upper_bound)
    """
    n_runs = len(all_run_fold_accuracies)

    # Calculate mean accuracy for each run (across its folds)
    run_mean_accuracies = []
    for i, run_fold_accs in enumerate(all_run_fold_accuracies):
        run_fold_counts = fold_sample_counts[i]

        assert len(run_fold_accs) == len(
            run_fold_counts
        ), "accuracy measures does not match num folds"

        # Weighted average based on fold sizes
        weighted_acc = calculate_weighted_run_average(run_fold_accs, run_fold_counts)
        run_mean_accuracies.append(weighted_acc)

    # Overall mean across runs
    overall_mean = np.mean(run_mean_accuracies)

    # Step 1: Calculate variance from run-to-run differences
    variance_between_runs = np.var(run_mean_accuracies, ddof=1)

    # Step 2: Calculate variance from finite sample size across all folds
    fold_variances = []
    for i, run_fold_accs in enumerate(all_run_fold_accuracies):
        run_fold_counts = fold_sample_counts[i]

        # Calculate binomial variance for each fold
        fold_vars = [
            acc * (1 - acc) / count
            for acc, count in zip(run_fold_accs, run_fold_counts)
        ]
        # Weight by sample count and average
        weighted_var = calculate_weighted_run_average(fold_vars, run_fold_counts)
        fold_variances.append(weighted_var)

    mean_variance_within = np.mean(fold_variances)

    # Step 3: Combine the variances
    total_variance = variance_between_runs + mean_variance_within

    # Step 4: Calculate confidence interval using t-distribution
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n_runs - 1)
    margin_error = t_critical * np.sqrt(total_variance / n_runs)

    lower_bound = overall_mean - margin_error
    upper_bound = overall_mean + margin_error

    return overall_mean, margin_error, lower_bound, upper_bound


def compare_model_accuracies(
    model_a_accuracies: list[float], model_b_accuracies: list[float], alpha=0.05
):
    """
    Perform paired t-test to compare the runs across two models

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


def compare_logo_cv_models(
    model_a_fold_accuracies: list[list[float]],
    model_b_fold_accuracies: list[list[float]],
    model_a_fold_counts: list[list[int]],
    model_b_fold_counts: list[list[int]],
    alpha=0.05,
):
    """
    Compare two models evaluated with LOGO-CV using the existing comparison function

    Parameters:
    model_a_fold_accuracies: List of lists, fold accuracies for each run of model A
    model_b_fold_accuracies: List of lists, fold accuracies for each run of model B
    model_a_fold_counts: Sample counts for each fold in model A
    model_b_fold_counts: Sample counts for each fold in model B
    alpha: Significance level (default: 0.05)

    Returns:
    tuple: (significant_difference, p_value, mean_difference)
    """
    # Ensure equal number of runs
    assert len(model_a_fold_accuracies) == len(
        model_b_fold_accuracies
    ), "Number of runs must match for both models"
    assert len(model_a_fold_counts) == len(
        model_b_fold_counts
    ), "Number of runs must match for both models"

    # Calculate weighted average accuracy for each run
    model_a_run_accuracies = []
    model_b_run_accuracies = []

    for i in range(len(model_a_fold_accuracies)):
        # Calculate weighted average for model A, run i
        weighted_acc_a = calculate_weighted_run_average(
            model_a_fold_accuracies[i], model_a_fold_counts[i]
        )
        model_a_run_accuracies.append(weighted_acc_a)

        # Same for model B
        weighted_acc_b = calculate_weighted_run_average(
            model_b_fold_accuracies[i], model_b_fold_counts[i]
        )
        model_b_run_accuracies.append(weighted_acc_b)

    return compare_model_accuracies(
        model_a_run_accuracies, model_b_run_accuracies, alpha
    )


if __name__ == "__main__":
    # Example usage
    model_1_accuracies = [0.82, 0.85, 0.83, 0.84, 0.86]
    mean, err, lower, upper = calculate_confidence_interval(model_1_accuracies)
    print(f"Model accuracy: {mean:.4f}+-{err:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")

    model_2_accuracies = [0.78, 0.80, 0.79, 0.81, 0.82]
    significant, p_value, mean_diff = compare_model_accuracies(
        model_1_accuracies, model_2_accuracies
    )
    print(
        f"Model comparison: Significant difference: {significant}, p-value: {p_value:.4f}, Mean difference: {mean_diff:.4f}"
    )
