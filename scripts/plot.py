import numpy as np
import matplotlib.pyplot as plt

# Original data
dataset_times = [9438.36, 810.04, 21.01, 258.17, 200.49, 198.07, 197.52, 198.35, 
                 201.85, 206.21, 206.54, 202.97, 201.86, 210.72, 233.08, 208.05, 
                 205.40, 205.85, 210.52, 230.77]
explicit_times = [8479.43, 793.44, 13.18, 271.39, 229.04, 226.91, 225.25, 224.38, 
                  224.29, 225.61, 222.07, 229.00, 225.85, 224.18, 228.21, 227.92, 
                  227.56, 228.55, 225.98, 227.24]

expl_mb = [440, 501, 518, 518, 518, 508, 508, 517, 518, 509, 518, 508]
laz_mb = [376, 375, 374, 374, 374, 374, 374, 374, 374, 374, 374, 374]

# Remove outliers using IQR
def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [x for x in data if lower <= x <= upper]

filtered_dataset = remove_outliers(dataset_times)
filtered_explicit = remove_outliers(explicit_times)

print(f"Explicit {np.mean(filtered_explicit)}, {np.std(filtered_explicit)}")
print(f"Lazy {np.mean(filtered_dataset)}, {np.std(filtered_dataset)}")


print(f"Explicit mb {np.mean(expl_mb)}, {np.std(expl_mb)}")
print(f"Lazy mb {np.mean(laz_mb)}, {np.std(laz_mb)}")
# Plot horizontal boxplot
plt.figure(figsize=(8, 4))
plt.boxplot([filtered_dataset, filtered_explicit],
            labels=['Lazy pipeline', 'Explicit pipeline'],
            notch=False, vert=False)
plt.xlabel('Time (ms)')
plt.ylabel('Pipeline')
# plt.title('Distribution of Pipeline Step Timings')
plt.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
plt.savefig("plot.png")
# plt.show()
