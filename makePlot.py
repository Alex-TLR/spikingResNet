import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'text.usetex': True,  # Use LaTeX for rendering text
    'font.family': 'serif',  # Set the base font to serif
    'font.serif': ['Times New Roman'],  # Set Times New Roman (if available)
    'mathtext.fontset': 'stix',  # Use the STIX math font family (similar to IEEE)
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
})

# 97.75 97.85 98.30 98.32 97.43   98.11 98.31 98.63 98.61 97.56  10.24 10.23 7.73 7.18 9.15
# 97.58 97.77 98.09 97.96 96.93   97.79 98.08 98.33 98.21 97.15  11.17 10.59 9.46 9.48 14.13
# 91.70 93.28 93.96 92.75 93.66   74.31 79.73 85.62 76.52 83.71  21.97 19.60 19.84 21.22 24.35


# Example data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.plot(x, y)
plt.title("AUROC")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")

# Save the plot as a PDF with embedded fonts
plt.savefig("ieee_plot.pdf", format="pdf", dpi=300, bbox_inches="tight", transparent=False)

# Close the plot
plt.close()