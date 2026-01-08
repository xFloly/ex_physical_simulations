import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from numba import njit
from scipy.sparse import coo_matrix, diags, csr_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# NUMBA: element forces
# ============================================================

@njit(cache=True)
def _compute_element_forces_numba(p0, p1, p2, x_inv, volume,
                                  young_modulus, poisson_ratio, use_cauchy):
    """
    Return 3x2 element force matrix for a single triangle.
    Uses: plane stress constitutive law, Cauchy or Green strain.
    """

    forces = np.zeros((3, 2), dtype=np.float64)
    if volume == 0.0:
        return forces

    # deformation gradient F = [p1-p0, p2-p0] * X_inv
    col0x = p1[0] - p0[0]
    col0y = p1[1] - p0[1]
    col1x = p2[0] - p0[0]
    col1y = p2[1] - p0[1]

    p00 = col0x * x_inv[0, 0] + col1x * x_inv[1, 0]
    p01 = col0x * x_inv[0, 1] + col1x * x_inv[1, 1]
    p10 = col0y * x_inv[0, 0] + col1y * x_inv[1, 0]
    p11 = col0y * x_inv[0, 1] + col1y * x_inv[1, 1]

    # grad(u) = F - I
    grad_u00 = p00 - 1.0
    grad_u11 = p11 - 1.0
    grad_u01 = p01
    grad_u10 = p10

    # strain
    if use_cauchy:
        strain00 = grad_u00
        strain11 = grad_u11
        strain01 = 0.5 * (grad_u01 + grad_u10)
    else:
        # Green strain: E = 0.5(grad+grad^T + grad^T grad)
        strain00 = grad_u00 + 0.5 * (grad_u00 * grad_u00 + grad_u10 * grad_u10)
        strain11 = grad_u11 + 0.5 * (grad_u01 * grad_u01 + grad_u11 * grad_u11)
        strain01 = 0.5 * (grad_u01 + grad_u10 + grad_u00 * grad_u01 + grad_u10 * grad_u11)

    # plane stress linear elasticity
    coeff = young_modulus / (1.0 - poisson_ratio * poisson_ratio)
    shear_coeff = coeff * (1.0 - poisson_ratio) * 0.5

    s0 = coeff * strain00 + coeff * poisson_ratio * strain11
    s1 = coeff * poisson_ratio * strain00 + coeff * strain11
    s2 = shear_coeff * strain01

    # reference gradients of shape functions
    grad1x = x_inv[0, 0]
    grad1y = x_inv[0, 1]
    grad2x = x_inv[1, 0]
    grad2y = x_inv[1, 1]
    grad0x = -(grad1x + grad2x)
    grad0y = -(grad1y + grad2y)

    grads = (
        (grad0x, grad0y),
        (grad1x, grad1y),
        (grad2x, grad2y),
    )

    # forces f_a = -Vol * sigma * grad(N_a)
    for idx in range(3):
        gradx, grady = grads[idx]
        fx = -(volume) * (s0 * gradx + s2 * grady)
        fy = -(volume) * (s2 * gradx + s1 * grady)
        forces[idx, 0] = fx
        forces[idx, 1] = fy

    return forces


@njit(cache=True)
def _assemble_forces_numba(positions, tri_indices, tri_x_inv, tri_volumes,
                          fixed_mask, young_modulus, poisson_ratio, use_cauchy,
                          apply_constraints):
    """Compute global internal force vector f(x)."""
    n = positions.shape[0]
    f_global = np.zeros((n, 2), dtype=np.float64)

    tri_count = tri_indices.shape[0]
    for tri_idx in range(tri_count):
        volume = tri_volumes[tri_idx]
        if volume == 0.0:
            continue

        i0, i1, i2 = tri_indices[tri_idx]

        elem_forces = _compute_element_forces_numba(
            positions[i0],
            positions[i1],
            positions[i2],
            tri_x_inv[tri_idx],
            volume,
            young_modulus,
            poisson_ratio,
            use_cauchy,
        )

        f_global[i0, 0] += elem_forces[0, 0]
        f_global[i0, 1] += elem_forces[0, 1]
        f_global[i1, 0] += elem_forces[1, 0]
        f_global[i1, 1] += elem_forces[1, 1]
        f_global[i2, 0] += elem_forces[2, 0]
        f_global[i2, 1] += elem_forces[2, 1]

    if apply_constraints:
        for i in range(n):
            if fixed_mask[i]:
                f_global[i, 0] = 0.0
                f_global[i, 1] = 0.0

    return f_global


# ============================================================
# FEM SOFT BODY (Implicit Euler)
# ============================================================

class FEMSoftBody:

    def __init__(self, young_modulus=15000, poisson_ratio=0.3, gravity=0.0,
                 fix_top=False, strain_type='green'):
        self.E = young_modulus
        self.nu = poisson_ratio
        self.gravity = gravity
        self.fix_top = fix_top
        self.strain_type = strain_type  # 'cauchy' or 'green'

        self.particles = []
        self.triangles = []

        self.positions = None
        self.positions0 = None
        self.velocities = None
        self.masses = None
        self.fixed_mask = None

        self.tri_indices_array = None
        self.tri_x_inv_array = None
        self.tri_volume_array = None

        self.last_force_matrix = None
        self.time = 0.0

        self.create_mesh()
        self.precompute_rest_config()

        # used to show rotation stability (torque impulse at start)
        self.rotation_impulse_time = 0.02

    def create_mesh(self):
        """Create a square grid mesh of triangles."""
        grid_size = 15
        spacing = 0.1
        offset_x = 2.0
        offset_y = 1.0

        self.particles = []
        for j in range(grid_size):
            for i in range(grid_size):
                x = (offset_x + i * spacing) / (spacing * grid_size)
                y = (offset_y + j * spacing) / (spacing * grid_size)
                pos = np.array([x, y], dtype=np.float64)
                self.particles.append({
                    'pos': pos.copy(),
                    'pos0': pos.copy(),
                    'vel': np.zeros(2, dtype=np.float64),
                    'mass': 1.0,
                    'fixed': (j == 0) and self.fix_top
                })

        self._initialize_particle_arrays()

        # Center and flip Y for nicer view
        centroid = self.positions.mean(axis=0)
        self.positions -= centroid
        self.positions0 -= centroid
        self.positions[:, 1] *= -1
        self.positions0[:, 1] *= -1

        self.ground_y = -2.0  # can disable collisions if you want
        self.left_wall = -10
        self.right_wall = 10

        # Build triangle connectivity
        self.triangles = []
        for j in range(grid_size - 1):
            for i in range(grid_size - 1):
                idx = j * grid_size + i
                self.triangles.append({'indices': [idx, idx + 1, idx + grid_size]})
                self.triangles.append({'indices': [idx + 1, idx + grid_size + 1, idx + grid_size]})

    def precompute_rest_config(self):
        """Compute X_inv and volumes for each triangle in rest configuration."""
        x_inv_list = []
        volume_list = []
        for tri in self.triangles:
            i0, i1, i2 = tri['indices']
            x0 = self.positions0[i0]
            x1 = self.positions0[i1]
            x2 = self.positions0[i2]

            X = np.column_stack([x1 - x0, x2 - x0])
            volume = 0.5 * abs(np.linalg.det(X))

            if volume < 1e-12:
                tri['X_inv'] = np.eye(2)
                tri['volume'] = 0.0
            else:
                tri['X_inv'] = np.linalg.inv(X)
                tri['volume'] = volume

            x_inv_list.append(tri['X_inv'])
            volume_list.append(tri['volume'])

        self.tri_indices_array = np.ascontiguousarray(
            np.array([tri['indices'] for tri in self.triangles], dtype=np.int64)
        )
        self.tri_x_inv_array = np.ascontiguousarray(np.array(x_inv_list, dtype=np.float64))
        self.tri_volume_array = np.ascontiguousarray(np.array(volume_list, dtype=np.float64))

    def assemble_internal_forces(self, apply_constraints=True):
        """Compute f_int(x) using Numba."""
        f_global = _assemble_forces_numba(
            self.positions,
            self.tri_indices_array,
            self.tri_x_inv_array,
            self.tri_volume_array,
            self.fixed_mask,
            self.E,
            self.nu,
            self.strain_type == 'cauchy',
            apply_constraints
        )
        return f_global

    # ============================================================
    # STIFFNESS MATRIX via Finite Differences: K = df/dx
    # ============================================================

    def compute_stiffness_fd(self, eps=1e-6):
        """
        Compute stiffness matrix K = df/dx using finite differences:
          K[:, j] = (f(x + eps e_j) - f(x)) / eps
        Returns:
          K (csr_matrix), f0_flat
        """

        n = self.positions.shape[0]
        dof = 2 * n

        # baseline force f0
        f0 = self.assemble_internal_forces(apply_constraints=False).reshape(-1)

        rows = []
        cols = []
        vals = []

        x_flat = self.positions.reshape(-1)

        for j in range(dof):
            # perturb
            x_flat[j] += eps
            f1 = self.assemble_internal_forces(apply_constraints=False).reshape(-1)
            x_flat[j] -= eps

            df = (f1 - f0) / eps

            # sparse insert
            for i in range(dof):
                v = df[i]
                if abs(v) > 1e-12:
                    rows.append(i)
                    cols.append(j)
                    vals.append(v)

        K = coo_matrix((vals, (rows, cols)), shape=(dof, dof)).tocsr()
        return K, f0

    # ============================================================
    # IMPLICIT EULER STEP:
    # (M - dt^2 K) dx = dt M v + dt^2 f
    # x_{t+1} = x_t + dx
    # v_{t+1} = dx / dt
    # ============================================================

    def update_implicit(self, dt, eps_K=1e-6):
        self.time += dt

        n = self.positions.shape[0]
        dof = 2 * n

        # internal forces + stiffness
        K, f_flat = self.compute_stiffness_fd(eps=eps_K)

        # add gravity (external) to RHS force vector:
        # f_total = f_internal + f_external
        # Here f_external is mass*gravity in y direction.
        f_total = f_flat.copy()
        for i in range(n):
            f_total[2*i + 1] -= self.masses[i] * self.gravity

        # initial torque impulse to get 180deg rotation
        if self.time <= self.rotation_impulse_time:
            center = self.positions.mean(axis=0)
            rotation_strength = 200.0
            for i in range(n):
                r = self.positions[i] - center
                tangential = np.array([-r[1], r[0]])
                f_total[2*i] += rotation_strength * self.masses[i] * tangential[0]
                f_total[2*i + 1] += rotation_strength * self.masses[i] * tangential[1]

        # diagonal mass matrix M
        M_diag = np.repeat(self.masses, 2)

        # A = M - dt^2 K
        A = diags(M_diag) - (dt * dt) * K

        # b = dt M v + dt^2 f_total
        v_flat = self.velocities.reshape(-1)
        b = dt * (M_diag * v_flat) + (dt * dt) * f_total

        # solve for dx
        dx = spsolve(A.tocsr(), b)

        # apply constraints
        fixed_dof = np.repeat(self.fixed_mask, 2)
        dx[fixed_dof] = 0.0

        # update x, v
        x_flat = self.positions.reshape(-1)
        x_new = x_flat + dx
        v_new = dx / dt

        self.positions[:] = x_new.reshape((n, 2))
        self.velocities[:] = v_new.reshape((n, 2))

        # keep force matrix for visualization (convert to Nx2)
        self.last_force_matrix = f_total.reshape((n, 2))

    def get_triangle_positions(self):
        return [self.positions[np.array(tri['indices'], dtype=np.int64)]
                for tri in self.triangles]

    def get_particle_positions(self):
        return self.positions.copy()

    def _initialize_particle_arrays(self):
        positions = np.array([p['pos'] for p in self.particles], dtype=np.float64)
        rest_positions = np.array([p['pos0'] for p in self.particles], dtype=np.float64)
        velocities = np.array([p['vel'] for p in self.particles], dtype=np.float64)
        masses = np.array([p['mass'] for p in self.particles], dtype=np.float64)
        fixed_mask = np.array([p['fixed'] for p in self.particles], dtype=bool)

        self.positions = np.ascontiguousarray(positions)
        self.positions0 = np.ascontiguousarray(rest_positions)
        self.velocities = np.ascontiguousarray(velocities)
        self.masses = np.ascontiguousarray(masses)
        self.fixed_mask = fixed_mask

        # keep dict references synced
        for idx, particle in enumerate(self.particles):
            particle['pos'] = self.positions[idx]
            particle['pos0'] = self.positions0[idx]
            particle['vel'] = self.velocities[idx]


# ============================================================
# VISUALIZATION / ANIMATION
# ============================================================

def main():
    sim = FEMSoftBody(
        young_modulus=15000,
        poisson_ratio=0.3,
        gravity=0.0,          # keep 0 to see pure rotation stability
        fix_top=False,
        strain_type='green',  # 'cauchy' or 'green'
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    initial_pos = sim.get_particle_positions()
    x_extent = np.max(np.abs(initial_pos[:, 0])) + 1.0
    y_extent = np.max(np.abs(initial_pos[:, 1])) + 1.0
    ax.set_xlim(-x_extent, x_extent)
    ax.set_ylim(-y_extent, y_extent)
    ax.set_title("2D FEM Soft Body Simulation (Implicit Euler + Stiffness Matrix)")

    patches = []
    for _ in sim.triangles:
        patch = Polygon([[0, 0], [0, 0], [0, 0]], fc='lightblue', ec='blue', alpha=0.6)
        ax.add_patch(patch)
        patches.append(patch)

    pts, = ax.plot([], [], 'o', color='darkblue', ms=4)
    fixed_pts, = ax.plot([], [], 'o', color='red', ms=6)

    zero_forces = np.zeros_like(initial_pos)
    force_quiver = ax.quiver(initial_pos[:, 0], initial_pos[:, 1],
                             zero_forces[:, 0], zero_forces[:, 1],
                             color='orange', angles='xy', scale_units='xy',
                             scale=1.0, width=0.003, alpha=0.7)

    txt = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    frame = [0]
    force_scale = 0.002

    def init():
        force_quiver.set_offsets(initial_pos)
        force_quiver.set_UVC(zero_forces[:, 0], zero_forces[:, 1])
        return patches + [pts, fixed_pts, force_quiver, txt]

    def animate(_):
        # implicit step doesn't need huge substeps
        dt = 0.001
        sim.update_implicit(dt, eps_K=2e-6)

        tri_pos = sim.get_triangle_positions()
        for patch, verts in zip(patches, tri_pos):
            patch.set_xy(verts)

        pos = sim.get_particle_positions()
        mask = sim.fixed_mask
        pts.set_data(pos[~mask, 0], pos[~mask, 1])
        fixed_pts.set_data(pos[mask, 0], pos[mask, 1])

        if sim.last_force_matrix is not None:
            forces = sim.last_force_matrix
            scaled_forces = forces * force_scale
            force_quiver.set_offsets(pos)
            force_quiver.set_UVC(scaled_forces[:, 0], scaled_forces[:, 1])

        frame[0] += 1
        txt.set_text(f"Frame: {frame[0]}  time={sim.time:.3f}s")
        return patches + [pts, fixed_pts, force_quiver, txt]

    anim = FuncAnimation(fig, animate, init_func=init, frames=500, interval=20, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
