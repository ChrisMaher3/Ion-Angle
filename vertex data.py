import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "2Pa 100w Gnd ar 15.csv"
df = pd.read_csv(INPUT_FILE)
df.columns = ["AR", "Flux"]
df["Angle_deg"] = np.degrees(np.arctan(1 / df["AR"]))

theta_theory = 8
FIRST_BIN = 4.399
edges_rest = np.linspace(FIRST_BIN, 90, 26)
angle_bins = np.concatenate(([0, FIRST_BIN], edges_rest[1:]))
bin_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])

raw_flux, _ = np.histogram(
    df["Angle_deg"],
    bins=angle_bins,
    weights=df["Flux"]
)

theta_rad = np.radians(bin_centers)
jacobian = 1 / (np.sin(theta_rad)**2)
corrected_flux = raw_flux * jacobian

normalized_flux = corrected_flux / np.sum(corrected_flux)

angles_full = np.concatenate((-bin_centers[::-1], [np.nan], bin_centers))
flux_full = np.concatenate((normalized_flux[::-1], [np.nan], normalized_flux))

plt.figure(figsize=(9, 5))
plt.plot(angles_full, flux_full, "-o", linewidth=2, markersize=4)
plt.xlabel("Angle (deg)")
plt.ylabel("Normalised Corrected Flux")
plt.title("Vertex 2Pa 100w rf")
plt.axvline(theta_theory, color="red", linestyle="--", linewidth=2, label="Theory")
plt.axvline(-theta_theory, color="red", linestyle="--", linewidth=2)
plt.legend()
plt.show()