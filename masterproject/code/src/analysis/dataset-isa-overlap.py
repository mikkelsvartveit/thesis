import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3


# Load the architecture columns from both CSVs
def get_architectures_cpu_rec():
    df = pd.read_csv("../../dataset/cpu_rec-features.csv", sep=";")
    names = []
    for _, row in df.iterrows():
        buildcross = str(row.get("buildcross_name", "")).strip()
        isa = str(row.get("isa_detect_name", "")).strip()
        arch = str(row.get("architecture", "")).strip()
        # Prefer buildcross_name, then isa_detect_name, then architecture
        if buildcross and buildcross.lower() != "nan" and buildcross.lower() != "none":
            names.append(buildcross)
        elif isa and isa.lower() != "nan" and isa.lower() != "none":
            names.append(isa)
        elif arch and arch.lower() != "nan" and arch.lower() != "none":
            names.append(arch)
    return set(names)


def get_architectures_buildcross():
    df = pd.read_csv("../../dataset/buildcross/labels.csv", sep=";")
    df.columns = df.columns.str.strip()
    archs = set(df["architecture"].dropna().astype(str).str.strip())
    return archs


def get_architectures_isadetect():
    df = pd.read_csv("../../dataset/ISAdetect-features.csv", sep=";")
    archs = set(df["architecture"].dropna().str.strip())
    return archs


def plot_venn():
    cpu_rec_archs = get_architectures_cpu_rec()
    isadetect_archs = get_architectures_isadetect()
    buildcross_archs = get_architectures_buildcross()
    plt.figure(figsize=(10, 8))
    v = venn3(
        [cpu_rec_archs, isadetect_archs, buildcross_archs],
        set_labels=("CpuRec", "ISAdetect", "Buildcross"),
        set_colors=("#8ecae6", "#ffd166", "#b7e4c7"),  # blue, yellow, green
        alpha=0.7,
    )
    # Set intersection color for all three
    if v.get_patch_by_id("111"):
        v.get_patch_by_id("111").set_color("#f7b7a3")  # pastel coral
        v.get_patch_by_id("111").set_alpha(0.8)
    # Add edge colors for clarity
    for patch in ["100", "010", "001", "110", "101", "011", "111"]:
        p = v.get_patch_by_id(patch)
        if p:
            p.set_edgecolor("#222")
            p.set_linewidth(1.5)
    plt.title("ISA Overlap: CpuRec vs ISAdetect vs Buildcross")
    plt.tight_layout()
    plt.savefig("output/dataset-isa-overlap.png")
    plt.show()


if __name__ == "__main__":
    plot_venn()
