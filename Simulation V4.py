import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# PLASMA ION SIMULATION - CCP ARGON
# Child-Langmuir sheath, self-consistent thickness, RF time-varying field
# Per-ion birth phase, leapfrog integration, high RF resolution timestep
# Collisionless - valid for pressure < 2 Pa
# No fixed seed - results vary each run
# =============================================================================

# --- USER INPUTS -------------------------------------------------------------
ion_mass      = 40.0      # amu
electron_temp = 2.0       # eV
ion_temp      = 0.025     # eV
n_e           = 1e17      # m^-3   plasma density (controls sheath thickness)
V_dc          = -150.0    # V      DC bias
V_rf          = 100.0     # V      RF amplitude
f_rf          = 13.56e6   # Hz     RF frequency
g1_v          = -75       # V      grid 1 voltage
pressure      = 0.5       # Pa     informational - collisionless assumed
n_particles   = 25_000    # ions to simulate

# --- CONSTANTS ---------------------------------------------------------------
electron = 1.60218e-19    # C
amu      = 1.66054e-27    # kg
eps0     = 8.85418e-12    # F/m

# --- DERIVED QUANTITIES ------------------------------------------------------
mass_kg   = ion_mass * amu
omega_rf  = 2 * np.pi * f_rf
rf_period = 1.0 / f_rf

# Velocities
v_bohm    = np.sqrt(electron_temp * electron / mass_kg)
v_thermal = np.sqrt(ion_temp * electron / mass_kg)
vx_sigma  = v_bohm + v_thermal       # lateral spread: Bohm + thermal

# Average sheath voltage => mean ion energy
V_avg     = abs(V_dc) + V_rf / 2.0
ey_energy = V_avg
vy_mean   = np.sqrt(2.0 * V_avg * electron / mass_kg)

# Self-consistent Child-Langmuir sheath thickness
lambda_D  = np.sqrt(eps0 * electron_temp * electron / (n_e * electron**2))
d_sheath  = (2.0/3.0) * lambda_D * (V_avg / electron_temp) ** 0.75

# Transit time and dimensionless ratio
transit_est   = d_sheath / (vy_mean / 2.0)
transit_ratio = transit_est / rf_period

# Timestep: high RF resolution (800 steps/cycle) but also resolves transit
dt = min(rf_period / 800.0, transit_est / 200.0)

# Expected IEDF peak positions
E_low_peak  = abs(V_dc) - V_rf / 2.0
E_high_peak = abs(V_dc) + V_rf / 2.0

print("=" * 60)
print("PLASMA ION SIMULATION - PARAMETERS")
print("=" * 60)
print(f"Ion mass:                {ion_mass} amu (Argon)")
print(f"Plasma density:          {n_e:.1e} m^-3")
print(f"Electron temp:           {electron_temp} eV")
print(f"Ion temp:                {ion_temp} eV")
print(f"Debye length:            {lambda_D*1e6:.1f} µm")
print(f"V_dc:                    {V_dc} V")
print(f"V_rf amplitude:          {V_rf} V")
print(f"RF frequency:            {f_rf/1e6:.2f} MHz  (T = {rf_period*1e9:.1f} ns)")
print(f"Avg sheath voltage:      {V_avg:.1f} V  =>  {ey_energy:.1f} eV mean energy")
print(f"Sheath thickness (C-L):  {d_sheath*1e3:.3f} mm")
print(f"Bohm velocity:           {v_bohm:.1f} m/s")
print(f"Ion thermal velocity:    {v_thermal:.1f} m/s")
print(f"Lateral sigma (vx):      {vx_sigma:.1f} m/s")
print(f"Mean forward velocity:   {vy_mean:.1f} m/s")
print(f"Est. transit time:       {transit_est*1e9:.1f} ns")
print(f"Transit / RF period:     {transit_ratio:.2f}  ", end="")
if transit_ratio < 0.5:
    print("(<< 1: well-separated IEDF peaks)")
elif transit_ratio < 2.0:
    print("(~ 1: moderate peak separation)")
else:
    print("(>> 1: peaks will merge, consider higher n_e)")
print(f"Timestep dt:             {dt*1e12:.1f} ps")
print(f"Expected IEDF peaks:     {E_low_peak:.1f} eV  and  {E_high_peak:.1f} eV")
print(f"Particles:               {n_particles:,}")
print("=" * 60)

# =============================================================================
# CHILD-LANGMUIR ELECTRIC FIELD
#
# Potential:  phi(z) = phi_wall * (z/d)^(4/3)
# E field:    E(z)   = -(4/3) * phi_wall/d * (z/d)^(1/3)
#
# Uses signed phi_wall so field direction is always correct:
# negative phi_wall => field points toward electrode => accelerates Ar+ ions
# =============================================================================
def cl_efield(z_pos, phi_wall, d):
    z_safe = np.clip(z_pos, 1e-9, d)
    return -(4.0/3.0) * (phi_wall / d) * (z_safe / d) ** (1.0/3.0)

# =============================================================================
# INITIALISE PARTICLES
# =============================================================================
rng = np.random.default_rng()   # no fixed seed - varies each run

# Each ion born at random RF phase
birth_phase = rng.uniform(0.0, 2.0 * np.pi, n_particles)

# Lateral velocity: Gaussian, sigma = Bohm + thermal
vx = rng.normal(0.0, vx_sigma, n_particles)

# Forward velocity: Bohm drift + thermal spread, always positive (sheath criterion)
vz = v_bohm + np.abs(rng.normal(0.0, v_thermal, n_particles))

# Starting position: sheath edge
z = np.full(n_particles, 1e-9)

# Tracking arrays
active           = np.ones(n_particles, dtype=bool)
impact_energy_eV = np.zeros(n_particles)
impact_angle_deg = np.zeros(n_particles)
impact_phase_out = np.zeros(n_particles)
n_steps_taken    = np.zeros(n_particles, dtype=int)

# =============================================================================
# MAIN TRAJECTORY LOOP - leapfrog integration
# Run until all particles have hit or max time exceeded
# =============================================================================
print("Running trajectory simulation...")

t         = 0.0
step      = 0
max_time  = rf_period * 30   # safety cap: 30 RF cycles

while np.any(active) and t < max_time:
    idx = np.where(active)[0]

    # Instantaneous wall voltage for each ion (per-ion birth phase)
    phi_wall = V_dc + V_rf * np.sin(birth_phase[idx] + omega_rf * t)

    # Child-Langmuir field and acceleration
    E = cl_efield(z[idx], phi_wall, d_sheath)
    a = (electron * E) / mass_kg

    # Leapfrog: velocity half-step, position full step
    vz[idx] += a * dt
    z[idx]  += vz[idx] * dt

    # --- Detect electrode impacts (z >= d_sheath) ---
    hit = active & (z >= d_sheath)
    if np.any(hit):
        hidx = np.where(hit)[0]
        speed_sq                  = vx[hidx]**2 + vz[hidx]**2
        impact_energy_eV[hidx]    = 0.5 * mass_kg * speed_sq / electron
        impact_angle_deg[hidx]    = np.degrees(np.arctan(vx[hidx] / vz[hidx]))
        impact_phase_out[hidx]    = np.mod(birth_phase[hidx] + omega_rf * t, 2*np.pi)
        n_steps_taken[hidx]       = step
        active[hit]               = False

    # --- Remove backscattered ions (z <= 0) ---
    escaped = active & (z <= 0)
    if np.any(escaped):
        active[escaped] = False

    t    += dt
    step += 1

    if step % 50000 == 0:
        print(f"  Step {step:,}  |  active: {np.sum(active):,}  |  "
              f"impacts: {np.sum(n_steps_taken > 0):,}  |  t = {t*1e9:.1f} ns")

# =============================================================================
# POST-PROCESS
# =============================================================================
hit_mask         = n_steps_taken > 0
impact_angle     = impact_angle_deg[hit_mask]
impact_energy_eV = impact_energy_eV[hit_mask]
n_impacts        = len(impact_angle)

print(f"\nSimulation complete!")
print(f"  Particles:  {n_particles:,}   Impacts: {n_impacts:,}  "
      f"({100*n_impacts/n_particles:.1f}%)   Steps: {step:,}")

sim_sigma = np.std(impact_angle)
sim_fwhm  = 2 * np.sqrt(2 * np.log(2)) * sim_sigma

print(f"\nAngle:   mean={np.mean(impact_angle):.3f}°  "
      f"std={sim_sigma:.3f}°  FWHM={sim_fwhm:.3f}°")
print(f"Energy:  mean={np.mean(impact_energy_eV):.1f} eV  "
      f"std={np.std(impact_energy_eV):.1f} eV  "
      f"range=[{np.min(impact_energy_eV):.1f}, {np.max(impact_energy_eV):.1f}] eV")

# =============================================================================
# ANALYTICAL GAUSSIAN OVERLAY (angle only)
# =============================================================================
sigma_analytical    = np.arctan(vx_sigma / vy_mean) * 180.0 / np.pi
fwhm_analytical     = 2 * np.sqrt(2 * np.log(2)) * sigma_analytical
angle_range         = np.linspace(-5*sigma_analytical, 5*sigma_analytical, 500)
analytical_gaussian = (1.0 / (sigma_analytical * np.sqrt(2*np.pi))) * \
                      np.exp(-0.5 * (angle_range / sigma_analytical)**2)

# =============================================================================
# PLOTTING
# =============================================================================
fig = plt.figure(figsize=(14, 10))


gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.42)

# --- Plot 1: Ion Angular Distribution Function (IADF) -----------------------
ax1 = fig.add_subplot(gs[0])
counts, bin_edges = np.histogram(impact_angle, bins=200, density=True)
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
ax1.bar(bin_centres, counts, width=np.diff(bin_edges),
        color='royalblue', alpha=0.6, label='Simulation (IADF)')
ax1.plot(angle_range, analytical_gaussian, color='red', linewidth=2,
         linestyle='--', label=f'Analytical Gaussian (FWHM={fwhm_analytical:.2f}°)')

sim_peak = np.max(counts)
ax1.axhline(y=sim_peak/2, color='gray', linestyle=':', linewidth=1.2,
            label=f'Sim FWHM = {sim_fwhm:.2f}°')
ax1.axvline(x=-sim_fwhm/2, color='gray', linestyle=':', linewidth=0.8)
ax1.axvline(x= sim_fwhm/2, color='gray', linestyle=':', linewidth=0.8)

ax1.set_xlabel('Impact Angle (degrees)', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title('Ion Angular Distribution Function (IADF)', fontsize=12)
ax1.legend(fontsize=10)


param_text = (
              f"Electron temp: {electron_temp} eV\n"
              f"Ion temp: {ion_temp} eV\n"
              f"V_dc: {V_dc} V\n"
              f"V_rf: {V_rf} V\n"
              f"RF Freq: {f_rf/1e6:.2f} MHz\n"
              f"Avg Ion Energy: {ey_energy:.1f} eV\n"
           )
ax1.text(0.01, 0.97, param_text, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                   edgecolor='gray', alpha=0.8))

# --- Plot 2: IEDF (line plot) ------------------------------------------------
ax2 = fig.add_subplot(gs[1])
counts_e, bins_e = np.histogram(impact_energy_eV, bins=200, density=True)
ax2.plot(bins_e[:-1], counts_e, color='teal', linewidth=2)

ax2.axvline(x=np.mean(impact_energy_eV), color='red', linestyle=':', linewidth=1.2,
            label=f'Mean = {np.mean(impact_energy_eV):.1f} eV')
ax2.set_xlabel('Ion Impact Energy (eV)', fontsize=12)
ax2.set_ylabel('Probability Density', fontsize=12)
ax2.set_title('Ion Energy Distribution Function (IEDF)', fontsize=12)
ax2.legend(fontsize=10)


plt.savefig('graph.png', dpi=150,
            bbox_inches='tight')
plt.show()

print("Plot saved.")