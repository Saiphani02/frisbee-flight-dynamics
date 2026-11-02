"""
Flight Dynamics of a Frisbee (Disc Golf Disc)
Student: Sai Phanidhar Ammiraju
Course: Flight Dynamics | Florida Atlantic University
Professor: Tom Lawrence

This script simulates the 3D flight dynamics of a frisbee/disc golf disc
using aerodynamic force models and numerical ODE integration.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os

# ─────────────────────────────────────────
# Physical & Aerodynamic Parameters
# ─────────────────────────────────────────
m   = 0.175       # mass [kg]  (standard disc golf disc)
R   = 0.107       # radius [m]
A   = np.pi * R**2  # planform area [m^2]
rho = 1.225       # air density [kg/m^3]  (sea level)
g   = 9.81        # gravity [m/s^2]

# Aerodynamic coefficients (empirically derived for disc golf discs)
CL0  =  0.10      # lift coefficient at zero angle of attack
CLa  =  1.80      # lift coefficient slope [1/rad]
CD0  =  0.08      # zero-lift drag coefficient
CDa  =  2.72      # induced drag coefficient [1/rad^2]
Cm0  =  0.01      # pitching moment coefficient at alpha=0
Cma  = -0.40      # pitching moment slope [1/rad]
Cmq  = -0.005     # pitch-damping coefficient

# Disc inertia moments
Izz  = 0.5  * m * R**2   # spin axis (axial)
Ixx  = 0.25 * m * R**2   # transverse (pitch/roll)


def frisbee_eom(t, state, spin_rate):
    """
    Equations of motion for a frisbee in 3D flight.

    State vector:
        [x, y, z, vx, vy, vz, phi, theta, dphi_dt, dtheta_dt]
    where phi   = roll angle  (bank)
          theta = pitch angle (angle of attack proxy)

    Returns dstate/dt
    """
    x, y, z, vx, vy, vz, phi, theta, dphi, dtheta = state

    # Velocity magnitude
    V = np.sqrt(vx**2 + vy**2 + vz**2)
    if V < 1e-6:
        V = 1e-6

    # Angle of attack (angle between disc plane and velocity vector)
    alpha = theta  # simplified: pitch angle drives AoA

    # Dynamic pressure
    q_dyn = 0.5 * rho * V**2

    # Aerodynamic coefficients
    CL = CL0 + CLa * alpha
    CD = CD0 + CDa * alpha**2

    # Lift force (perpendicular to velocity, in vertical plane)
    L = q_dyn * A * CL
    D = q_dyn * A * CD

    # Unit vectors
    v_hat = np.array([vx, vy, vz]) / V

    # Lift direction: perpendicular to velocity in the plane of disc normal
    # Simplified: lift acts upward (z-direction component modified by bank)
    lift_x = -L * (vz / V) * np.cos(phi)
    lift_y = -L * (vz / V) * np.sin(phi)
    lift_z =  L * np.sqrt(vx**2 + vy**2) / V * np.cos(phi)

    # Drag force opposes velocity
    drag_x = -D * vx / V
    drag_y = -D * vy / V
    drag_z = -D * vz / V

    # Total accelerations
    ax = (lift_x + drag_x) / m
    ay = (lift_y + drag_y) / m
    az = (lift_z + drag_z) / m - g

    # Gyroscopic precession: spin produces a gyroscopic torque
    # When the disc pitches, gyroscopic moment causes roll (and vice versa)
    L_spin = Izz * spin_rate  # angular momentum from spin

    # Pitching moment
    M_pitch = q_dyn * A * (2 * R) * (Cm0 + Cma * alpha + Cmq * (dtheta * R / V))

    # Gyroscopic precession rate: Ω_precess = torque / (Izz * spin_rate)
    # Pitch torque causes roll precession
    d2phi    = (M_pitch - L_spin * dtheta) / Ixx
    d2theta  = (M_pitch + L_spin * dphi)  / Ixx

    return [vx, vy, vz, ax, ay, az, dphi, dtheta, d2phi, d2theta]


def simulate(v0=20.0, angle_deg=15.0, spin_rpm=600, side_angle_deg=0.0):
    """
    Simulate frisbee flight.

    Parameters
    ----------
    v0            : launch speed [m/s]
    angle_deg     : launch elevation angle [degrees]
    spin_rpm      : spin rate [RPM]
    side_angle_deg: lateral launch angle [degrees]

    Returns
    -------
    t, x, y, z arrays
    """
    angle  = np.radians(angle_deg)
    side   = np.radians(side_angle_deg)
    spin   = spin_rpm * 2 * np.pi / 60  # convert to rad/s

    vx0 = v0 * np.cos(angle) * np.cos(side)
    vy0 = v0 * np.cos(angle) * np.sin(side)
    vz0 = v0 * np.sin(angle)

    # Initial state: [x,y,z, vx,vy,vz, phi, theta, dphi, dtheta]
    state0 = [0, 0, 1.5,    # start 1.5 m above ground (release height)
              vx0, vy0, vz0,
              0.0, angle,    # initial pitch = launch angle
              0.0, 0.0]

    # Integrate until disc hits ground (z = 0) or t_max
    t_max = 20.0

    def hit_ground(t, y, spin): return y[2]
    hit_ground.terminal  = True
    hit_ground.direction = -1

    sol = solve_ivp(frisbee_eom, [0, t_max], state0, args=(spin,),
                    events=hit_ground, max_step=0.02,
                    rtol=1e-6, atol=1e-8)

    return sol.t, sol.y[0], sol.y[1], sol.y[2]


# ─────────────────────────────────────────
# Run Simulations
# ─────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(out_dir, exist_ok=True)

plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})

# ── Plot 1: Effect of Launch Angle ──────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Effect of Launch Angle on Frisbee Trajectory\n(v₀ = 20 m/s, Spin = 600 RPM)", fontsize=14, fontweight='bold')

angles = [5, 10, 15, 20, 25, 30]
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(angles)))

for ang, col in zip(angles, colors):
    t, x, y, z = simulate(v0=20, angle_deg=ang, spin_rpm=600)
    z_clip = np.maximum(z, 0)
    axes[0].plot(x, z_clip, color=col, linewidth=2, label=f'{ang}°')
    axes[1].plot(x, z_clip, color=col, linewidth=2, label=f'{ang}°')

axes[0].set_xlabel("Horizontal Distance (m)")
axes[0].set_ylabel("Height (m)")
axes[0].set_title("Trajectory Profiles")
axes[0].legend(title="Launch Angle", loc='upper right')
axes[0].set_xlim(left=0); axes[0].set_ylim(bottom=0)
axes[0].grid(True, alpha=0.3)

# Range vs Angle
ranges = []
for ang in np.linspace(1, 40, 60):
    t, x, y, z = simulate(v0=20, angle_deg=ang, spin_rpm=600)
    ranges.append(x[-1])
angles_sweep = np.linspace(1, 40, 60)
axes[1].cla()
axes[1].plot(angles_sweep, ranges, 'b-', linewidth=2.5)
axes[1].set_xlabel("Launch Angle (degrees)")
axes[1].set_ylabel("Range (m)")
axes[1].set_title("Range vs. Launch Angle")
axes[1].grid(True, alpha=0.3)
optimal_idx = np.argmax(ranges)
axes[1].axvline(angles_sweep[optimal_idx], color='red', linestyle='--', alpha=0.7,
                label=f"Optimal: {angles_sweep[optimal_idx]:.1f}°")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{out_dir}/plot1_launch_angle.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 1 saved.")

# ── Plot 2: Effect of Spin Rate ──────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Effect of Spin Rate on Frisbee Flight\n(v₀ = 20 m/s, Launch Angle = 15°)", fontsize=14, fontweight='bold')

spins = [200, 400, 600, 800, 1000, 1200]
colors2 = plt.cm.plasma(np.linspace(0.1, 0.9, len(spins)))

for spin, col in zip(spins, colors2):
    t, x, y, z = simulate(v0=20, angle_deg=15, spin_rpm=spin)
    z_clip = np.maximum(z, 0)
    axes[0].plot(x, z_clip, color=col, linewidth=2, label=f'{spin} RPM')

axes[0].set_xlabel("Horizontal Distance (m)")
axes[0].set_ylabel("Height (m)")
axes[0].set_title("Trajectory vs. Spin Rate")
axes[0].legend(title="Spin Rate", loc='upper right')
axes[0].set_xlim(left=0); axes[0].set_ylim(bottom=0)
axes[0].grid(True, alpha=0.3)

# Range vs Spin Rate
spin_sweep = np.linspace(100, 1500, 60)
spin_ranges = []
for sp in spin_sweep:
    t, x, y, z = simulate(v0=20, angle_deg=15, spin_rpm=sp)
    spin_ranges.append(x[-1])

axes[1].plot(spin_sweep, spin_ranges, 'purple', linewidth=2.5)
axes[1].set_xlabel("Spin Rate (RPM)")
axes[1].set_ylabel("Range (m)")
axes[1].set_title("Range vs. Spin Rate")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/plot2_spin_rate.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 2 saved.")

# ── Plot 3: Effect of Launch Speed ──────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Effect of Launch Speed on Frisbee Flight\n(Launch Angle = 15°, Spin = 600 RPM)", fontsize=14, fontweight='bold')

speeds = [10, 14, 18, 22, 26, 30]
colors3 = plt.cm.cool(np.linspace(0.1, 0.9, len(speeds)))

for spd, col in zip(speeds, colors3):
    t, x, y, z = simulate(v0=spd, angle_deg=15, spin_rpm=600)
    z_clip = np.maximum(z, 0)
    axes[0].plot(x, z_clip, color=col, linewidth=2, label=f'{spd} m/s')

axes[0].set_xlabel("Horizontal Distance (m)")
axes[0].set_ylabel("Height (m)")
axes[0].set_title("Trajectory vs. Launch Speed")
axes[0].legend(title="Launch Speed", loc='upper right')
axes[0].set_xlim(left=0); axes[0].set_ylim(bottom=0)
axes[0].grid(True, alpha=0.3)

speed_sweep = np.linspace(5, 35, 60)
speed_ranges = []
for spd in speed_sweep:
    t, x, y, z = simulate(v0=spd, angle_deg=15, spin_rpm=600)
    speed_ranges.append(x[-1])

axes[1].plot(speed_sweep, speed_ranges, 'green', linewidth=2.5)
axes[1].set_xlabel("Launch Speed (m/s)")
axes[1].set_ylabel("Range (m)")
axes[1].set_title("Range vs. Launch Speed")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/plot3_launch_speed.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 3 saved.")

# ── Plot 4: 3D Trajectory with lateral drift ──────
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle("3D Frisbee Trajectory — Effect of Lateral Launch Angle\n(v₀ = 22 m/s, 15° elevation, 600 RPM)", fontsize=13, fontweight='bold')

laterals = [-10, -5, 0, 5, 10]
colors4 = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(laterals)))

for lat, col in zip(laterals, colors4):
    t, x, y, z = simulate(v0=22, angle_deg=15, spin_rpm=600, side_angle_deg=lat)
    z_clip = np.maximum(z, 0)
    ax.plot(x, y, z_clip, color=col, linewidth=2, label=f'{lat}°')

ax.set_xlabel("X — Downrange (m)")
ax.set_ylabel("Y — Lateral (m)")
ax.set_zlabel("Z — Height (m)")
ax.legend(title="Lateral Angle")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out_dir}/plot4_3d_trajectory.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 4 saved.")

# ── Plot 5: Summary Dashboard ────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Frisbee Flight Dynamics — Summary Dashboard\nFlight Dynamics Project | Sai Phanidhar Ammiraju | FAU",
             fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# 1) Canonical trajectory
ax1 = fig.add_subplot(gs[0, 0])
t, x, y, z = simulate(v0=22, angle_deg=15, spin_rpm=700)
z_clip = np.maximum(z, 0)
ax1.plot(x, z_clip, 'b-', linewidth=2.5)
ax1.fill_between(x, 0, z_clip, alpha=0.15, color='blue')
ax1.set_xlabel("Range (m)"); ax1.set_ylabel("Height (m)")
ax1.set_title("Reference Trajectory\n(22 m/s, 15°, 700 RPM)")
ax1.grid(True, alpha=0.3)

# 2) Range vs launch angle
ax2 = fig.add_subplot(gs[0, 1])
angles_s = np.linspace(1, 40, 80)
rng_a = [simulate(v0=20, angle_deg=a, spin_rpm=600)[1][-1] for a in angles_s]
ax2.plot(angles_s, rng_a, 'darkgreen', linewidth=2)
opt_idx = int(np.argmax(rng_a))
ax2.axvline(angles_s[opt_idx], color='red', linestyle='--', alpha=0.8,
            label=f"Opt: {angles_s[opt_idx]:.1f}°")
ax2.set_xlabel("Launch Angle (°)"); ax2.set_ylabel("Range (m)")
ax2.set_title("Range vs. Launch Angle"); ax2.legend(); ax2.grid(True, alpha=0.3)

# 3) Range vs spin
ax3 = fig.add_subplot(gs[0, 2])
spins_s = np.linspace(100, 1400, 60)
rng_s = [simulate(v0=20, angle_deg=15, spin_rpm=sp)[1][-1] for sp in spins_s]
ax3.plot(spins_s, rng_s, 'purple', linewidth=2)
ax3.set_xlabel("Spin Rate (RPM)"); ax3.set_ylabel("Range (m)")
ax3.set_title("Range vs. Spin Rate"); ax3.grid(True, alpha=0.3)

# 4) Range vs launch speed
ax4 = fig.add_subplot(gs[1, 0])
speeds_s = np.linspace(5, 35, 60)
rng_sp = [simulate(v0=v, angle_deg=15, spin_rpm=600)[1][-1] for v in speeds_s]
ax4.plot(speeds_s, rng_sp, 'darkorange', linewidth=2)
ax4.set_xlabel("Launch Speed (m/s)"); ax4.set_ylabel("Range (m)")
ax4.set_title("Range vs. Launch Speed"); ax4.grid(True, alpha=0.3)

# 5) Lift & Drag vs AoA
ax5 = fig.add_subplot(gs[1, 1])
alphas = np.linspace(-10, 30, 200) * np.pi / 180
q_dyn_ref = 0.5 * rho * 20**2
CL_vals = CL0 + CLa * alphas
CD_vals = CD0 + CDa * alphas**2
L_vals = q_dyn_ref * A * CL_vals
D_vals = q_dyn_ref * A * CD_vals
ax5.plot(np.degrees(alphas), L_vals, 'b-', linewidth=2, label='Lift (N)')
ax5.plot(np.degrees(alphas), D_vals, 'r-', linewidth=2, label='Drag (N)')
ax5.set_xlabel("Angle of Attack (°)"); ax5.set_ylabel("Force (N)")
ax5.set_title("Lift & Drag vs. AoA\n(V = 20 m/s)")
ax5.legend(); ax5.grid(True, alpha=0.3)
ax5.axvline(0, color='k', linewidth=0.5)

# 6) L/D ratio
ax6 = fig.add_subplot(gs[1, 2])
LD = np.where(np.abs(CD_vals) > 1e-6, CL_vals / CD_vals, np.nan)
ax6.plot(np.degrees(alphas), LD, 'teal', linewidth=2)
ax6.set_xlabel("Angle of Attack (°)"); ax6.set_ylabel("L/D Ratio")
ax6.set_title("Lift-to-Drag Ratio vs. AoA")
ax6.grid(True, alpha=0.3)
ax6.axhline(0, color='k', linewidth=0.5)

plt.savefig(f"{out_dir}/plot5_dashboard.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 5 (Dashboard) saved.")

print("\n✓ All simulations complete. Plots saved to:", out_dir)

# ── Print key results ────────────────────
print("\n── Key Results ──")
t, x, y, z = simulate(v0=22, angle_deg=15, spin_rpm=700)
print(f"  Reference flight (22 m/s, 15°, 700 RPM): Range = {x[-1]:.1f} m, Flight time = {t[-1]:.2f} s")
opt_ang = angles_s[int(np.argmax(rng_a))]
print(f"  Optimal launch angle (20 m/s, 600 RPM): {opt_ang:.1f}°")
t2, x2, y2, z2 = simulate(v0=20, angle_deg=opt_ang, spin_rpm=600)
print(f"  Maximum range at optimal angle: {x2[-1]:.1f} m")
