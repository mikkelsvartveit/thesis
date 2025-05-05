import matplotlib.pyplot as plt
import os

data_string = """
arm64 Accuracy: 99.97%
ppc64 Accuracy: 100.00%
ia64 Accuracy: 100.00%
i386 Accuracy: 99.98%
sh4 Accuracy: 100.00%
sparc64 Accuracy: 99.79%
mips Accuracy: 100.00%
s390x Accuracy: 100.00%
alpha Accuracy: 100.00%
m68k Accuracy: 100.00%
x32 Accuracy: 100.00%
s390 Accuracy: 100.00%
armel Accuracy: 100.00%
hppa Accuracy: 100.00%
riscv64 Accuracy: 100.00%
powerpc Accuracy: 100.00%
mipsel Accuracy: 100.00%
armhf Accuracy: 100.00%
ppc64el Accuracy: 100.00%
amd64 Accuracy: 100.00%
powerpcspe Accuracy: 100.00%
sparc Accuracy: 100.00%
mips64el Accuracy: 100.00%
"""

# Parse the data
data = []
for line in data_string.strip().split("\n"):
    if line:
        parts = line.split(" Accuracy: ")
        arch = parts[0]
        accuracy_str = parts[1].strip("%")
        accuracy = float(accuracy_str) / 100.0
        data.append((arch, accuracy))

# Sort data by accuracy (ascending)
sorted_data = sorted(data, key=lambda item: item[1])

# Separate architectures and accuracies for plotting
architectures = [item[0] for item in sorted_data]
accuracies = [item[1] for item in sorted_data]

# Create the horizontal bar chart
plt.figure(figsize=(9, 12))  # Use similar figure size as the original script
plt.barh(architectures, accuracies, color="yellowgreen")  # Use similar color

plt.xlabel("Accuracy")
plt.title("Accuracy by ISA")  # Add a title for clarity
plt.xlim(0, 1)  # Set x-axis limit from 0 to 1

plt.tight_layout()

# Save the figure
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, "isadetect-traintest-accuracy-by-isa.png"))

print(
    f"Bar chart saved to {os.path.join(output_dir, 'isadetect-traintest-accuracy-by-isa.png')}"
)
