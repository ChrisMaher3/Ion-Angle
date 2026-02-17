import numpy as np
import matplotlib.pyplot as plt

# inputs
ion_mass = 40
electron_temp = 2
ey_energy = 150
ion_temp = 0.025
g1_c = 3e-3
g1_g2 = 0.005
g2_v = 3000
g1_v = -75

# constants
electron = 1.602e-19
amu = 1.6726e-27
spacer = 1e-4
boltzmann = 1.38e-23

# outputs
bohm = np.sqrt((electron_temp * electron / (amu * ion_mass)))
vy_intial = np.sqrt((2 * ey_energy * electron) / (ion_mass * amu))
ion_vel = np.sqrt((2 * ion_temp * electron) / (ion_mass * amu))
vy_col = np.sqrt((2 * (ey_energy - g1_v) * electron) / (ion_mass * amu))

# angles
half_angle_spread = np.arctan((bohm + ion_temp) / vy_intial) * 180 / np.pi
fwhm = half_angle_spread
sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
resolution = 0.02
start = -5 * sigma
stop = 5 * sigma
angles = np.arange(start, stop, resolution)
amplitude = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (angles / sigma) ** 2)

# plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(angles, amplitude, color='royalblue', linewidth=2)

ax.axhline(y=max(amplitude) / 2, color='red', linestyle='--', linewidth=1.2, label=f'FWHM = {fwhm:.2f}°')
ax.axvline(x=-fwhm/2, color='gray', linestyle=':', linewidth=1)
ax.axvline(x= fwhm/2, color='gray', linestyle=':', linewidth=1)

param_text = (
    f"Electron Temp: {electron_temp} eV\n"
    f"Ion Energy (Ey): {ey_energy} eV\n"
    f"Ion Mass: {ion_mass} amu\n"
    f"Ion Temp: {ion_temp} eV\n"
    f"G1 Voltage: {g1_v} V"
)
ax.text(0.02, 0.97, param_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

ax.set_xlabel('Angle (degrees)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_title('Theortical Ion Angle Distribution', fontsize=14)
ax.legend(fontsize=11)

plt.tight_layout()
plt.show()