import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

INPUT_FILE = "2pa argon 02 10-3 35 vdc.csv"
df = pd.read_csv(INPUT_FILE)
df.columns = ["AR", "Flux"]
df["Angle_deg"] = np.degrees(np.arctan(1 / df["AR"]))

print("Plasma Potential")
PlasmaP = int(input())
top_vel = (PlasmaP * 1.60217653e-19) / 3.321e-26
Vx = np.sqrt(top_vel)
theta_theory = np.degrees(np.arctan(2196.42 / Vx))
print(f"Theory angle: {theta_theory:.2f}°")

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

flux_smoothed = gaussian_filter1d(normalized_flux, sigma=2)

centre_value = max(flux_smoothed[0], flux_smoothed[1])
angles_full = np.concatenate((-bin_centers[::-1], [0], bin_centers))
flux_smoothed_full = np.concatenate((flux_smoothed[::-1], [centre_value], flux_smoothed))

# Half-angle measurement
peak = np.nanmax(flux_smoothed_full)
half_max = peak / 2
pos_mask = ~np.isnan(angles_full) & (angles_full >= 0)
pos_angles = angles_full[pos_mask]
pos_flux = flux_smoothed_full[pos_mask]
interp_func = interp1d(pos_flux[::-1], pos_angles[::-1])
half_angle = float(interp_func(half_max))
measured_fwhm = 2 * half_angle

print(f"Measured half-angle: {half_angle:.2f}°")
print(f"Measured FWHM: {measured_fwhm:.2f}°")

plt.figure(figsize=(9, 5))
plt.plot(angles_full, flux_smoothed_full, "-o", linewidth=2, markersize=4)
plt.xlabel("Angle (deg)")
plt.ylabel("Normalised Corrected Flux")
plt.title("Vertex 2Pa, Argon/o2 20:3, 35 Vdc, Flt")
#plt.axvline(theta_theory, color="red", linestyle="--", linewidth=2,
         #   label=f"Theory ({theta_theory:.1f}°)")
#plt.axvline(-theta_theory, color="red", linestyle="--", linewidth=2)
plt.axvline(half_angle, color="green", linestyle=":", linewidth=1.5,
            label=f"Half-width ({half_angle:.1f}°)")
plt.axvline(-half_angle, color="green", linestyle=":", linewidth=1.5)
plt.legend()
plt.tight_layout()
plt.savefig("vertex_argon02IADF.png", dpi=900)
plt.show()