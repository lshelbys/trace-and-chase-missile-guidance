"""
Trace And Chase - 3D Missile-Aircraft Pursuit Simulation
=========================================================
Improvements over original:
  - SimConfig dataclass consolidates all parameters (no scattered globals)
  - TrajectoryGenerator class replaces fragile global-state pattern
  - Pure pursuit guidance (correct "Trace and Chase" law) — restored after PN bug
  - Closing-speed HUD readout (pre-computed, no dt-scaling bug)
  - Mach number display
  - Intercept flash effect
  - Dark theme with improved contrast
  - Removed unused imports (solve_ivp, interp1d, time)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# CONFIGURATION  — edit everything here, nowhere else
# ============================================================================
@dataclass
class SimConfig:
    # ---- Flight segments ----
    straight_time_1: float = 25.0
    curve_time:      float = 25.0
    straight_time_2: float = 25.0
    tmax:            float = 75.0
    dt:              float = 0.001

    # ---- Aircraft ----
    targ_vel:           float = 750.0
    turn_angle:         float = -np.pi * 4 / 3   # rad  (−240°)
    yz_angle:           float = -np.pi / 12       # rad
    climb_rate_curve:   float = -0.001
    aircraft_start_loc: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 12000.0])
    )

    # ---- Missile ----
    miss_vel:            float = 800.0
    missile_start_loc:   np.ndarray = field(
        default_factory=lambda: np.array([13000.0, 12000.0, 0.0])
    )
    missile_launch_time: float = 0.0
    kill_dist:           float = 2.0

    # ---- Animation ----
    animation_interval: int = 5     # ms between frames
    max_anim_frames:    int = 600


# ============================================================================
# TRAJECTORY GENERATOR
# ============================================================================
class TrajectoryGenerator:
    """Generates aircraft positions step-by-step with no global state."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

        if abs(cfg.turn_angle) < 1e-9:
            self.radius = 1e9
        else:
            self.radius = (cfg.targ_vel * cfg.curve_time) / cfg.turn_angle

        self._curve_pos:    Optional[np.ndarray] = None
        self._curve_center: Optional[np.ndarray] = None
        self._seg2_pos:     Optional[np.ndarray] = None

    def reset(self):
        self._curve_pos    = None
        self._curve_center = None
        self._seg2_pos     = None

    def position_at(self, t: float, prev_pos: Optional[np.ndarray]) -> np.ndarray:
        cfg = self.cfg

        # ---- Segment 1: straight in +X ----
        if t <= cfg.straight_time_1:
            return np.array([
                cfg.aircraft_start_loc[0] + cfg.targ_vel * t,
                cfg.aircraft_start_loc[1],
                cfg.aircraft_start_loc[2],
            ])

        # ---- Segment 2: curved turn ----
        elif t <= cfg.straight_time_1 + cfg.curve_time:
            if self._curve_pos is None:
                self._curve_pos = (
                    prev_pos.copy() if prev_pos is not None
                    else np.array([
                        cfg.aircraft_start_loc[0] + cfg.targ_vel * cfg.straight_time_1,
                        cfg.aircraft_start_loc[1],
                        cfg.aircraft_start_loc[2],
                    ])
                )
                self._curve_center = np.array([
                    self._curve_pos[0],
                    self._curve_pos[1] + self.radius * np.cos(cfg.yz_angle),
                    self._curve_pos[2] + self.radius * np.sin(cfg.yz_angle),
                ])

            tc        = t - cfg.straight_time_1
            arc_angle = -np.pi / 2 + tc * cfg.turn_angle / cfg.curve_time
            climb     = (cfg.targ_vel ** 2
                         * (1.0 - np.cos(np.pi * tc / cfg.curve_time))
                         * cfg.climb_rate_curve)

            cx, cy, cz = self._curve_center
            return np.array([
                cx + self.radius * np.cos(arc_angle),
                cy + self.radius * np.sin(arc_angle) * np.cos(cfg.yz_angle)
                   + np.cos(cfg.yz_angle + np.pi / 2) * climb,
                cz + self.radius * np.sin(arc_angle) * np.sin(cfg.yz_angle)
                   + np.sin(cfg.yz_angle + np.pi / 2) * climb,
            ])

        # ---- Segment 3: straight after turn ----
        elif t <= cfg.straight_time_1 + cfg.curve_time + cfg.straight_time_2:
            if self._seg2_pos is None:
                self._seg2_pos = (
                    prev_pos.copy() if prev_pos is not None
                    else (self._curve_pos.copy() if self._curve_pos is not None
                          else cfg.aircraft_start_loc.copy())
                )

            ts = t - (cfg.straight_time_1 + cfg.curve_time)
            d  = self._post_turn_direction()
            return self._seg2_pos + cfg.targ_vel * ts * d

        # ---- Beyond end: freeze at last position ----
        else:
            if self._seg2_pos is not None:
                d = self._post_turn_direction()
                return self._seg2_pos + cfg.targ_vel * cfg.straight_time_2 * d
            return np.array([
                cfg.aircraft_start_loc[0] + cfg.targ_vel * cfg.straight_time_1,
                cfg.aircraft_start_loc[1],
                cfg.aircraft_start_loc[2],
            ])

    def _post_turn_direction(self) -> np.ndarray:
        ta, ya = self.cfg.turn_angle, self.cfg.yz_angle
        return np.array([np.cos(ta),
                         np.sin(ta) * np.cos(ya),
                         np.sin(ta) * np.sin(ya)])


# ============================================================================
# SIMULATION RUNNER
# ============================================================================
def run_simulation(cfg: SimConfig):
    times    = np.arange(0.0, cfg.tmax, cfg.dt)
    n        = len(times)

    # ---- Aircraft trajectory ----
    gen = TrajectoryGenerator(cfg)
    target_states = np.empty((n, 3))
    target_states[0] = cfg.aircraft_start_loc.copy()
    for i in range(1, n):
        target_states[i] = gen.position_at(times[i], target_states[i - 1])

    # ---- Missile trajectory: pure pursuit ("Trace and Chase") ----
    missile_states = np.empty((n, 3))
    missile_states[0] = cfg.missile_start_loc.copy()

    missile_launched = False
    intercepted      = False
    intercept_index  = None
    intercept_time   = None

    for i in range(1, n):
        t = times[i]

        if t >= cfg.missile_launch_time and not missile_launched:
            missile_launched = True
            print(f"  Missile launched at t = {t:.3f} s")

        if not missile_launched:
            missile_states[i] = cfg.missile_start_loc.copy()
            continue

        if intercepted:
            missile_states[i] = missile_states[i - 1].copy()
            continue

        direction = target_states[i] - missile_states[i - 1]
        distance  = np.linalg.norm(direction)

        if distance <= cfg.kill_dist:
            intercept_index = i
            intercept_time  = t
            intercepted     = True
            print(f"  INTERCEPT at t = {t:.3f} s  |  miss = {distance:.3f} m")
            missile_states[i] = missile_states[i - 1].copy()
            continue

        if distance > 0:
            missile_states[i] = missile_states[i - 1] + (direction / distance) * cfg.miss_vel * cfg.dt
        else:
            missile_states[i] = missile_states[i - 1].copy()

    if not intercepted:
        final_miss = np.linalg.norm(target_states[-1] - missile_states[-1])
        print(f"  No intercept — final miss distance: {final_miss:.1f} m")

    return times, target_states, missile_states, intercept_index, intercept_time


# ============================================================================
# ANIMATION
# ============================================================================
def animate(cfg, times, target_states, missile_states, intercept_index):
    fig = plt.figure(figsize=(15, 10))
    ax  = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#0a0a14')
    ax.set_facecolor('#0a0a14')

    # ---- Axis limits ----
    all_pts    = np.vstack([target_states, missile_states])
    x_c = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2
    y_c = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2
    z_c = (all_pts[:, 2].max() + all_pts[:, 2].min()) / 2
    half_range = max(
        all_pts[:, 0].max() - all_pts[:, 0].min(),
        all_pts[:, 1].max() - all_pts[:, 1].min(),
        all_pts[:, 2].max() - all_pts[:, 2].min(),
    ) / 2 * 1.15

    ax.set_xlim(x_c - half_range, x_c + half_range)
    ax.set_ylim(y_c - half_range, y_c + half_range)
    ax.set_zlim(z_c - half_range, z_c + half_range)
    ax.set_box_aspect([1, 1, 1])

    # ---- Style ----
    ax.tick_params(colors='#aab', labelsize=8)
    ax.xaxis.label.set_color('#aab')
    ax.yaxis.label.set_color('#aab')
    ax.zaxis.label.set_color('#aab')
    ax.title.set_color('white')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Trace And Chase — 3D Pure Pursuit Simulation',
                 fontsize=13, fontweight='bold', pad=14)
    ax.grid(True, color='#223', linewidth=0.5)
    ax.view_init(elev=20, azim=45)

    # ---- Static markers ----
    ax.scatter(*target_states[0],  c='#00ff88', s=120, marker='s', zorder=5, label='Aircraft Start')
    ax.scatter(*missile_states[0], c='#ffaa00', s=120, marker='^', zorder=5, label='Missile Start')
    if intercept_index is not None:
        ax.scatter(*target_states[intercept_index], c='yellow', s=300,
                   marker='*', zorder=6,
                   label=f'Intercept  t={times[intercept_index]:.1f} s')

    # ---- Animated artists ----
    tgt_pt,    = ax.plot([], [], [], 'o', color='#4af', markersize=10, zorder=5)
    tgt_trail, = ax.plot([], [], [], '-', color='#4af', linewidth=1.8, alpha=0.55)
    mis_pt,    = ax.plot([], [], [], 'o', color='#f44', markersize=8,  zorder=5)
    mis_trail, = ax.plot([], [], [], '-', color='#f44', linewidth=1.4, alpha=0.55)
    flash_pt,  = ax.plot([], [], [], '*', color='yellow', markersize=28, zorder=7, alpha=0.0)

    # ---- HUD ----
    def txt(x, y, color, size=9):
        return ax.text2D(x, y, '', transform=ax.transAxes,
                         color=color, fontsize=size, fontfamily='monospace')

    hud_t   = txt(0.02, 0.97, 'white', 11)
    hud_spd = txt(0.02, 0.93, '#4af')
    hud_mis = txt(0.02, 0.89, '#f44')
    hud_sep = txt(0.02, 0.85, '#fa4')
    hud_cls = txt(0.02, 0.81, '#af4')
    hud_hit = ax.text2D(0.30, 0.50, '', transform=ax.transAxes,
                        color='yellow', fontsize=20, fontweight='bold',
                        fontfamily='monospace', alpha=0.0)

    # ---- Legend ----
    patches = [
        mpatches.Patch(color='#4af',    label='Aircraft'),
        mpatches.Patch(color='#f44',    label='Missile'),
        mpatches.Patch(color='#00ff88', label='Aircraft Start'),
        mpatches.Patch(color='#ffaa00', label='Missile Start'),
    ]
    if intercept_index is not None:
        patches.append(mpatches.Patch(color='yellow', label='Intercept'))
    ax.legend(handles=patches, loc='upper right',
              facecolor='#111', edgecolor='#334', labelcolor='white', fontsize=9)

    # ---- Pre-compute separation & closing speed arrays ----
    sep_all   = np.linalg.norm(target_states - missile_states, axis=1)
    close_spd = np.empty(len(times))
    close_spd[0] = 0.0
    close_spd[1:] = -(sep_all[1:] - sep_all[:-1]) / cfg.dt  # +ve = closing

    # ---- Frame list ----
    n    = len(times)
    skip = max(1, n // cfg.max_anim_frames)
    frames = list(range(0, n, skip))

    # ---- Init ----
    def init():
        for a in (tgt_pt, tgt_trail, mis_pt, mis_trail, flash_pt):
            a.set_data([], [])
            a.set_3d_properties([])
        for h in (hud_t, hud_spd, hud_mis, hud_sep, hud_cls, hud_hit):
            h.set_text('')
        return tgt_pt, tgt_trail, mis_pt, mis_trail, flash_pt, \
               hud_t, hud_spd, hud_mis, hud_sep, hud_cls, hud_hit

    # ---- Update ----
    def update(frame):
        # Aircraft
        tgt_pt.set_data([target_states[frame, 0]], [target_states[frame, 1]])
        tgt_pt.set_3d_properties([target_states[frame, 2]])
        tgt_trail.set_data(target_states[:frame+1, 0], target_states[:frame+1, 1])
        tgt_trail.set_3d_properties(target_states[:frame+1, 2])

        # Missile
        mis_pt.set_data([missile_states[frame, 0]], [missile_states[frame, 1]])
        mis_pt.set_3d_properties([missile_states[frame, 2]])
        mis_trail.set_data(missile_states[:frame+1, 0], missile_states[:frame+1, 1])
        mis_trail.set_3d_properties(missile_states[:frame+1, 2])

        # Intercept flash (fades over 1.5 sim-seconds)
        if intercept_index is not None and frame >= intercept_index:
            elapsed = (frame - intercept_index) * cfg.dt
            if elapsed < 1.5:
                alpha = max(0.0, 1.0 - elapsed / 1.5)
                flash_pt.set_data([target_states[intercept_index, 0]],
                                  [target_states[intercept_index, 1]])
                flash_pt.set_3d_properties([target_states[intercept_index, 2]])
                flash_pt.set_alpha(alpha)
                hud_hit.set_text('★  INTERCEPT  ★')
                hud_hit.set_alpha(alpha)
            else:
                flash_pt.set_alpha(0.0)
                hud_hit.set_text('')
        else:
            flash_pt.set_alpha(0.0)
            hud_hit.set_text('')

        # HUD values
        sep = sep_all[frame]
        cs  = close_spd[frame]
        spd = (np.linalg.norm(target_states[frame] - target_states[frame - 1]) / cfg.dt
               if frame > 0 else cfg.targ_vel)

        hud_t.set_text(  f'T = {times[frame]:6.2f} s')
        hud_spd.set_text(f'Aircraft speed : {spd:7.1f} m/s  ({spd/340:.2f} Mach)')
        hud_mis.set_text(f'Missile  speed : {cfg.miss_vel:7.1f} m/s  ({cfg.miss_vel/340:.2f} Mach)')
        hud_sep.set_text(f'Separation     : {sep:9.1f} m')
        hud_cls.set_text(f'Closing speed  : {cs:+9.1f} m/s')

        return (tgt_pt, tgt_trail, mis_pt, mis_trail, flash_pt,
                hud_t, hud_spd, hud_mis, hud_sep, hud_cls, hud_hit)

    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         blit=False, interval=cfg.animation_interval, repeat=True)

    print(f"\n  Animation: {len(frames)} frames  (skip={skip})  — close window to exit")
    plt.tight_layout()
    plt.show()
    return anim


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    cfg = SimConfig()

    print("=" * 60)
    print("  Trace And Chase — 3D Missile-Aircraft Pursuit Simulation")
    print("=" * 60)
    print(f"  Aircraft start : {cfg.aircraft_start_loc}")
    print(f"  Missile  start : {cfg.missile_start_loc}")
    print(f"  Aircraft speed : {cfg.targ_vel} m/s  ({cfg.targ_vel/340:.2f} Mach)")
    print(f"  Missile  speed : {cfg.miss_vel} m/s  ({cfg.miss_vel/340:.2f} Mach)")
    print(f"  Kill radius    : {cfg.kill_dist} m")
    print(f"  Turn angle     : {np.degrees(cfg.turn_angle):.1f}°")
    print(f"  Sim duration   : {cfg.tmax} s   dt = {cfg.dt} s")
    print("-" * 60)
    print("  Generating trajectories …")

    times, target_states, missile_states, intercept_index, intercept_time = \
        run_simulation(cfg)

    print("-" * 60)
    print("  Starting animation …")
    _anim = animate(cfg, times, target_states, missile_states, intercept_index)
