import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
import os


def font_context():
    # Define font paths
    font_dir = "../../../text/assets/fonts"
    font_regular = os.path.join(font_dir, "SourceSerif4-Regular.ttf")
    font_bold = os.path.join(font_dir, "SourceSerif4-Bold.ttf")
    font_semibold = os.path.join(font_dir, "SourceSerif4-SemiBold.ttf")

    # Register fonts with matplotlib
    for font_path in [font_regular, font_bold, font_semibold]:
        fm.fontManager.addfont(font_path)

    # Create font names
    font_family = "Source Serif 4"
    return plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": [font_family],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelweight": "regular",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
        }
    )


def model_swarmplot(
    model_names: list,
    model_accuracies: list[list[float]],
    model_95s: list[float],
    target_feature: str,
    file_name: str,
    **kwargs,
):
    model_acc_dict = {name: accs for name, accs in zip(model_names, model_accuracies)}
    # with font_context():
    palette = sns.color_palette("deep", n_colors=len(model_names))
    sns.set_palette(palette)

    # Create strip plot
    plt.figure(figsize=(9, 6))
    plt.ylabel("Model Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)

    args = {"data": model_acc_dict, "size": 4, **kwargs}
    sns.swarmplot(**args)

    # Add mean and 95% confidence intervals as error bars
    for i, model in enumerate(model_acc_dict.keys()):
        mean = sum(model_accuracies[i]) / len(model_accuracies[i])
        conf_95 = model_95s[i]
        plt.errorbar(
            x=i + 0.3,
            y=mean,
            yerr=conf_95,
            fmt="o",
            color="black",
            capsize=5,
            markersize=4,
            label=f"Mean ± 95% Confidence Interval" if i == 0 else None,
        )

    plt.legend(loc="lower left")
    plt.savefig(file_name, dpi=200, bbox_inches="tight")


def plot_model_comparison_table(model_comparisons, title, alpha=0.05, figsize=(15, 7)):
    """
    Create a visualization of model comparison results using matplotlib.

    Parameters:
    -----------
    model_comparisons : dict of dicts
        Dictionary where each key is a model name and each value is a dictionary
        mapping to comparison results with other models
    alpha : float
        Significance level for bolding p-values
    figsize : tuple
        Figure size
    """
    # Extract model names
    model_names = list(model_comparisons.keys())
    n_models = len(model_names)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Hide axes
    ax.axis("off")
    ax.axis("tight")

    # Create data for the table
    table_data = []
    for row_model in model_names:
        row_data = []
        for col_model in model_names:
            if row_model == col_model:
                # Diagonal cell
                row_data.append("---")
            else:
                comparison = model_comparisons[row_model][col_model]
                if comparison is None:
                    row_data.append("N/A")
                else:
                    # Unpack comparison results
                    is_significant, p_value, mean_diff, test_type = comparison

                    # Format the result
                    if mean_diff < 0:
                        direction = f"{col_model} < {row_model}"
                    else:
                        direction = f"{col_model} > {row_model}"

                    if is_significant:
                        if p_value < 0.001:
                            sig_mark = "***"  # p < 0.001
                        elif p_value < 0.01:
                            sig_mark = "**"  # p < 0.01
                        else:
                            sig_mark = "*"  # p < 0.05
                    else:
                        sig_mark = ""

                    result = f"Δ = {abs(mean_diff):.4f}\np = {p_value:.4f}{sig_mark} ({test_type})\n({direction})"
                    row_data.append(result)
        table_data.append(row_data)

    # Create the table
    table = ax.table(
        cellText=table_data,
        rowLabels=model_names,
        colLabels=model_names,
        loc="center",
        cellLoc="center",
        bbox=[0.076, 0.17, 0.9, 0.75],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.8, 3)  # Adjust height of rows

    # Style diagonal cells differently
    for i in range(n_models):
        cell = table[(i + 1, i)]
        cell.set_facecolor("#e8e8e8")

    # Style cells with significant results
    for i, row_model in enumerate(model_names):
        for j, col_model in enumerate(model_names):
            if row_model != col_model:
                comparison = model_comparisons[row_model][col_model]
                if comparison is not None:
                    is_significant, p_value, mean_diff, _ = comparison
                    if is_significant:
                        cell = table[(i + 1, j)]
                        # Make cell text bold for significant results
                        # cell.get_text().set_weight("bold")

                        if mean_diff > 0:  # col_model better than row_model
                            cell.set_facecolor("#d6f5d6")  # light green
                        else:
                            cell.set_facecolor("#f5d6d6")  # light red

    # Add a title
    plt.title(title, fontsize=14, fontweight="bold")

    plt.subplots_adjust(top=20)

    # Add footer with notes
    plt.figtext(
        0.1,
        0.01,
        """
        Notes:
        - Δ: Average accuracy difference between models
        - H_0: There is no significant difference in average accuracy between the models.
        - *: p < 0.05, **: p < 0.01, ***: p < 0.001
        - (w): Wilcoxon signed-rank test, (t): Paired t-test
        - Direction shows which model performs better
        - Green cells indicate the model in the column is significantly better than the model in the row
        - Red cells indicate the model in the row is significantly better than the model in the column
        """,
        ha="left",
        fontsize=8,
    )

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm,
    class_names,
    title="Confusion Matrix",
    cmap=plt.cm.RdYlGn,
    normalize=False,
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
        # Create custom annotations with % symbol
        annotations = [[f"{val:.1f}%" for val in row] for row in cm * 100]
    else:
        annotations = True

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm * 100 if normalize else cm,
        annot=annotations,
        fmt="",  # Empty format since we're providing custom annotations
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0 if normalize else None,
        vmax=100 if normalize else None,
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="black",
        annot_kws={"size": 13},
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
