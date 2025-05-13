import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def model_swarmplot(
    model_names: list,
    model_accuracies: list[list[float]],
    model_95s: list[float],
    target_feature: str,
    file_name: str,
    **kwargs,
):
    model_acc_dict = {name: accs for name, accs in zip(model_names, model_accuracies)}
    palette = sns.color_palette("deep", n_colors=len(model_names))
    sns.set_palette(palette)

    # Create strip plot
    plt.figure(figsize=(9, 7))
    plt.title(f"Swarmplot of {target_feature} accuracies")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
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
            label=f"Mean ± 95% CI" if i == 0 else None,
        )

    plt.legend(loc="lower left")
    plt.savefig(file_name, dpi=200, bbox_inches="tight")
