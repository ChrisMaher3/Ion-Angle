import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from plasmapy.particles import Particle
from plasmapy.formulary.lengths import Debye_length
from plasmapy.formulary.frequencies import plasma_frequency

# ============================================================================
# SECTION 1: PLASMA AND RF PARAMETERS
# ============================================================================
ion = Particle("Ar+")
m_i = ion.mass.to(u.kg).value
q_i = ion.charge.to(u.C).value
e = 1.602e-19

neutral = Particle("Ar")
m_n = neutral.mass.to(u.kg).value
T_gas = 300

pressure = 7  # mTorr
k_B = 1.381e-23
pressure_Pa = pressure * 0.133322
n_neutral = pressure_Pa / (k_B * T_gas)

print(f"Pressure = {pressure} mTorr = {pressure_Pa:.2f} Pa")
print(f"Neutral density = {n_neutral:.2e} m^-3")

sigma_collision = 5e-19
lambda_mfp = 1 / (n_neutral * sigma_collision)
print(f"Mean free path = {lambda_mfp*1000:.2f} mm")

# RF PARAMETERS
V_dc = -100.0
V_rf = 100.0
f_rf = 13.56e6
omega_rf = 2 * np.pi * f_rf

# PLASMA PARAMETERS
T_e = 3.0
n_e_val = 1e16

T_e_unit = T_e * u.eV
n_e = n_e_val / u.m**3

lambda_D = Debye_length(T_e_unit, n_e)
omega_pi = plasma_frequency(n_e, ion)
f_pi = omega_pi / (2 * np.pi * u.rad)
f_pi = f_pi.to(u.MHz)

V_sheath = abs(V_dc) + V_rf / 2
d_sheath_CL = (2/3) * lambda_D.value * (V_sheath / T_e)**(3/4)

# Use slightly larger sheath (more realistic)
d_sheath = 3e-3  # 3 mm
print(f"\nChild-Langmuir sheath estimate: {d_sheath_CL*1000:.2f} mm")
print(f"Using sheath thickness: {d_sheath*1000:.1f} mm")
print(f"Mean free path / sheath thickness = {lambda_mfp/d_sheath:.2f}")

print(f"\nDebye length = {lambda_D:.2e}")
print(f"Ion plasma frequency = {f_pi:.2f}")
print(f"RF frequency = {f_rf/1e6:.2f} MHz")

T_e_joules = T_e * e
v_bohm = np.sqrt(T_e_joules / m_i)
print(f"Bohm velocity = {v_bohm:.1f} m/s")

# Better transit time estimate
v_final_max = np.sqrt(v_bohm**2 + 2*q_i*(abs(V_dc)+V_rf/2)/m_i)
v_final_min = np.sqrt(v_bohm**2 + 2*q_i*(abs(V_dc)-V_rf/2)/m_i) if (abs(V_dc)-V_rf/2) > 0 else v_bohm
v_avg_max = (v_bohm + v_final_max) / 2
v_avg_min = (v_bohm + v_final_min) / 2
transit_time_min = d_sheath / v_avg_max
transit_time_max = d_sheath / v_avg_min

print(f"Expected transit time range: {transit_time_min*1e6:.2f} - {transit_time_max*1e6:.2f} µs")
print(f"RF periods during transit: {transit_time_min/(1/f_rf):.1f} - {transit_time_max/(1/f_rf):.1f}")

# ============================================================================
# SECTION 2: SIMULATION SETUP - CONTINUOUS INJECTION
# ============================================================================
n_particles_total = 10000
dt = 5e-11
t_end = 50e-6
steps = int(t_end / dt)

# Initialize arrays for particle data
z = np.array([])
v_z = np.array([])
v_x = np.array([])
v_y = np.array([])
particle_ids = np.array([])
particle_birth_time = np.array([])
particle_birth_phase = np.array([])

impact_energies = []
impact_angles = []
impact_times = []
impact_phases = []

n_collisions = 0
collision_positions = []

# Injection parameters
particles_injected = 0
injection_rate = 200
injection_interval = 500

track_particle_ids = [0, 50, 100]
track_z = {pid: [] for pid in track_particle_ids}
track_t = []

# ============================================================================
# COLLISION FUNCTIONS
# ============================================================================
def calculate_collision_probability(v_relative, dt):
    collision_rate = v_relative * n_neutral * sigma_collision
    prob = 1 - np.exp(-collision_rate * dt)
    return np.minimum(prob, 0.5)

def apply_collision(vx, vy, vz):
    v_th_neutral = np.sqrt(3 * k_B * T_gas / m_n)
    speed_new = np.random.rayleigh(v_th_neutral)
    theta = np.arccos(2 * np.random.random() - 1)
    phi = 2 * np.pi * np.random.random()

    vx_new = speed_new * np.sin(theta) * np.cos(phi)
    vy_new = speed_new * np.sin(theta) * np.sin(phi)
    vz_new = speed_new * np.cos(theta)

    return vx_new, vy_new, vz_new

# ============================================================================
# SECTION 3: MAIN SIMULATION LOOP
# ============================================================================
print(f"\nRunning simulation with continuous injection...")
print(f"Expected collisions per ion = {d_sheath/lambda_mfp:.2f}")

for step in range(steps):
    t = step * dt
    rf_phase = np.mod(omega_rf * t, 2*np.pi)

    # -----------------------------------------------------------------------
    # INJECT NEW PARTICLES
    # -----------------------------------------------------------------------
    if step % injection_interval == 0 and particles_injected < n_particles_total:
        n_inject = min(injection_rate, n_particles_total - particles_injected)
        z_new = np.ones(n_inject) * 0.01e-3
        v_thermal = v_bohm * 0.3
        v_z_new = np.random.normal(v_bohm * 0.5, v_thermal, n_inject)
        v_x_new = np.random.normal(0, v_thermal, n_inject)
        v_y_new = np.random.normal(0, v_thermal, n_inject)

        ids_new = np.arange(particles_injected, particles_injected + n_inject)
        birth_time_new = np.ones(n_inject) * t
        birth_phase_new = np.ones(n_inject) * rf_phase

        z = np.concatenate([z, z_new])
        v_z = np.concatenate([v_z, v_z_new])
        v_x = np.concatenate([v_x, v_x_new])
        v_y = np.concatenate([v_y, v_y_new])
        particle_ids = np.concatenate([particle_ids, ids_new])
        particle_birth_time = np.concatenate([particle_birth_time, birth_time_new])
        particle_birth_phase = np.concatenate([particle_birth_phase, birth_phase_new])

        particles_injected += n_inject

    if len(z) == 0:
        continue

    # -----------------------------------------------------------------------
    # ELECTRIC FIELD AND ACCELERATION
    # -----------------------------------------------------------------------
    V_rf_electrode = V_dc + V_rf * np.sin(rf_phase)
    E_z = -V_rf_electrode / d_sheath
    a_z = q_i * E_z / m_i

    v_z += a_z * dt
    z += v_z * dt

    # -----------------------------------------------------------------------
    # COLLISIONS
    # -----------------------------------------------------------------------
    speed = np.sqrt(v_x**2 + v_y**2 + v_z**2)
    collision_prob = calculate_collision_probability(speed, dt)
    random_numbers = np.random.random(len(z))
    collides = random_numbers < collision_prob

    if np.any(collides):
        n_coll = np.sum(collides)
        n_collisions += n_coll
        collision_positions.extend(z[collides])
        for idx in np.where(collides)[0]:
            v_x[idx], v_y[idx], v_z[idx] = apply_collision(v_x[idx], v_y[idx], v_z[idx])

    # -----------------------------------------------------------------------
    # TRACK PARTICLES
    # -----------------------------------------------------------------------
    if step % 2000 == 0:
        track_t.append(t * 1e6)
        for pid in track_particle_ids:
            mask = particle_ids == pid
            if np.any(mask):
                idx = np.where(mask)[0][0]
                track_z[pid].append(z[idx] * 1000)
            else:
                track_z[pid].append(None)

    # -----------------------------------------------------------------------
    # DETECT IMPACTS
    # -----------------------------------------------------------------------
    hit = z >= d_sheath
    if np.any(hit):
        vx, vy, vz = v_x[hit], v_y[hit], v_z[hit]
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        energy = 0.5 * m_i * speed**2 / e
        angle = np.degrees(np.arccos(np.clip(np.abs(vz)/speed, 0, 1)))

        impact_energies.extend(energy)
        impact_angles.extend(angle)
        impact_times.extend([t] * np.sum(hit))
        impact_phases.extend([rf_phase] * np.sum(hit))

        mask = ~hit
        z, v_z, v_x, v_y = z[mask], v_z[mask], v_x[mask], v_y[mask]
        particle_ids = particle_ids[mask]
        particle_birth_time = particle_birth_time[mask]
        particle_birth_phase = particle_birth_phase[mask]

    # Remove escaped particles
    escape = z <= 0
    if np.any(escape):
        mask = ~escape
        z, v_z, v_x, v_y = z[mask], v_z[mask], v_x[mask], v_y[mask]
        particle_ids = particle_ids[mask]
        particle_birth_time = particle_birth_time[mask]
        particle_birth_phase = particle_birth_phase[mask]

    if step % 100000 == 0:
        print(f"Step {step}/{steps} ({t*1e6:.2f} µs), active: {len(z)}, "
              f"injected: {particles_injected}, impacts: {len(impact_energies)}")

# ============================================================================
# RESULTS AND VISUALIZATION
# ============================================================================
print(f"\nSimulation complete!")
print(f"Total particles injected: {particles_injected}")
print(f"Total impacts: {len(impact_energies)}")
print(f"Total collisions: {n_collisions}")

if len(impact_energies) > 0:
    impact_energies = np.array(impact_energies)
    impact_angles = np.array(impact_angles)
    impact_times = np.array(impact_times)
    impact_phases = np.array(impact_phases)

    print(f"\nAverage collisions per ion: {n_collisions/len(impact_energies):.2f}")
    print(f"\nEnergy statistics:")
    print(f"  Mean: {np.mean(impact_energies):.1f} eV")
    print(f"  Median: {np.median(impact_energies):.1f} eV")
    print(f"  Std: {np.std(impact_energies):.1f} eV")
    print(f"  Min: {np.min(impact_energies):.1f} eV")
    print(f"  Max: {np.max(impact_energies):.1f} eV")

    print(f"\nAngle statistics:")
    print(f"  Mean: {np.mean(impact_angles):.1f}°")
    print(f"  Median: {np.median(impact_angles):.1f}°")
    print(f"  Std: {np.std(impact_angles):.1f}°")

# Visualization: Energy & Angle Distributions
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 5, wspace=0.3)

# Energy distribution (IEDF as line plot)
    ax1 = fig.add_subplot(gs[0, :3])
    counts, bins = np.histogram(impact_energies, bins=100, density=True)
    ax1.plot(bins[:-1], counts, color='teal', linewidth=2)
    ax1.axvline(np.mean(impact_energies), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(impact_energies):.1f} eV')
    ax1.axvline(abs(V_dc), color='orange', linestyle=':', linewidth=2,
                label=f'|V_dc|: {abs(V_dc):.0f} eV')
    ax1.set_xlabel("Ion Impact Energy (eV)", fontsize=12)
    ax1.set_ylabel("Probability Density", fontsize=12)
    ax1.set_title(f"Ion Energy Distribution Function (IEDF)\n"
                  f"{pressure} mTorr, d={d_sheath*1000:.1f} mm, n={len(impact_energies)}",
                  fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Angle distribution (histogram)
    ax2 = fig.add_subplot(gs[0, 3:])
    ax2.hist(impact_angles, bins=60, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(impact_angles), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(impact_angles):.1f}°')
    ax2.set_xlabel("Impact Angle (°)", fontsize=12)
    ax2.set_ylabel("Counts", fontsize=12)
    ax2.set_title("Angle Distribution", fontsize=13)
    ax2.set_xlim(0, 90)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Super title for the figure
    plt.suptitle(f"RF Plasma Ion Bombardment Simulation\nAr plasma, {pressure} mTorr, "
                 f"V_dc={V_dc}V, V_rf={V_rf}V, f={f_rf/1e6:.1f}MHz",
                 fontsize=14, fontweight='bold')
    
    plt.show()