"""
3D single-phase permeability prediction in porous media using MRT-LBM.

This example provides reusable geometry builders and lets you choose pressure
boundary-condition flavor ("zouhe" or "regularized") at runtime.
"""

import argparse
import os

import numpy as np

from src.boundary_conditions import BounceBack, Regularized, ZouHe
from src.lattice import LatticeD3Q19
from src.models import MRTSim
from src.utils import save_fields_vtk


def d3q19_mrt_matrix():
    e = LatticeD3Q19().c.T
    en = np.linalg.norm(e, axis=1)
    M = np.zeros((19, 19))
    M[0, :] = en**0
    M[1, :] = 19 * en**2 - 30
    M[2, :] = (21 * en**4 - 53 * en**2 + 24) / 2
    M[3, :] = e[:, 0]
    M[4, :] = (5 * en**2 - 9) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (5 * en**2 - 9) * e[:, 1]
    M[7, :] = e[:, 2]
    M[8, :] = (5 * en**2 - 9) * e[:, 2]
    M[9, :] = 3 * e[:, 0] ** 2 - en**2
    M[10, :] = (3 * en**2 - 5) * (3 * e[:, 0] ** 2 - en**2)
    M[11, :] = e[:, 1] ** 2 - e[:, 2] ** 2
    M[12, :] = (3 * en**2 - 5) * (e[:, 1] ** 2 - e[:, 2] ** 2)
    M[13, :] = e[:, 0] * e[:, 1]
    M[14, :] = e[:, 1] * e[:, 2]
    M[15, :] = e[:, 0] * e[:, 2]
    M[16, :] = (e[:, 1] ** 2 - e[:, 2] ** 2) * e[:, 0]
    M[17, :] = (e[:, 2] ** 2 - e[:, 0] ** 2) * e[:, 1]
    M[18, :] = (e[:, 0] ** 2 - e[:, 1] ** 2) * e[:, 2]
    return M


def build_sphere_pack_mask(nx, ny, nz, seed=11, n_spheres=120, r_min=2, r_max=5, target_porosity=0.7):
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    mask = np.zeros((nx, ny, nz), dtype=bool)

    for _ in range(n_spheres):
        cx = rng.integers(4, nx - 4)
        cy = rng.integers(4, ny - 4)
        cz = rng.integers(4, nz - 4)
        r = rng.integers(r_min, r_max + 1)
        sphere = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r**2
        mask |= sphere

    mask[0, :, :] = False
    mask[-1, :, :] = False

    current_porosity = 1.0 - np.mean(mask)
    if current_porosity < target_porosity:
        n_open = int((target_porosity - current_porosity) * nx * ny * nz)
        solid_cells = np.column_stack(np.where(mask))
        if n_open > 0 and solid_cells.shape[0] > 0:
            picked = solid_cells[rng.choice(solid_cells.shape[0], size=min(n_open, solid_cells.shape[0]), replace=False)]
            mask[picked[:, 0], picked[:, 1], picked[:, 2]] = False

    return mask


def build_cylinder_pack_mask(nx, ny, nz, seed=7, n_cylinders=40, r_min=2, r_max=4):
    rng = np.random.default_rng(seed)
    x, y, _ = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    mask = np.zeros((nx, ny, nz), dtype=bool)
    for _ in range(n_cylinders):
        cx = rng.integers(4, nx - 4)
        cy = rng.integers(4, ny - 4)
        r = rng.integers(r_min, r_max + 1)
        mask |= ((x - cx) ** 2 + (y - cy) ** 2 <= r**2)

    mask[0, :, :] = False
    mask[-1, :, :] = False
    return mask


class PorousPermeabilityMRT3D(MRTSim):
    def __init__(self, *args, solid_mask, rho0, delta_rho, omega, pressure_bc, **kwargs):
        self.solid_mask = solid_mask
        self.rho0 = rho0
        self.delta_rho = delta_rho
        self.omega = omega
        self.pressure_bc = pressure_bc
        super().__init__(*args, **kwargs)

    def set_boundary_conditions(self):
        inlet = self.boundingBoxIndices["left"]
        outlet = self.boundingBoxIndices["right"]

        rho_inlet = self.rho0 * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        rho_outlet = (self.rho0 - self.delta_rho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)

        bc_cls = ZouHe if self.pressure_bc == "zouhe" else Regularized
        self.BCs.append(bc_cls(tuple(inlet.T), self.gridInfo, self.precisionPolicy, "pressure", rho_inlet))
        self.BCs.append(bc_cls(tuple(outlet.T), self.gridInfo, self.precisionPolicy, "pressure", rho_outlet))

        solid_indices = np.column_stack(np.where(self.solid_mask))
        side_walls = np.concatenate(
            (
                self.boundingBoxIndices["bottom"],
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["front"],
                self.boundingBoxIndices["back"],
            )
        )
        walls = np.unique(np.concatenate((solid_indices, side_walls)), axis=0)
        self.BCs.append(BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy))

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho"][0, ...])
        u = np.array(kwargs["u"][0, ...])
        timestep = kwargs["timestep"]

        fluid = ~self.solid_mask
        u_mean = np.mean(u[..., 0][fluid])

        cs2 = 1.0 / 3.0
        nu = (1.0 / self.omega - 0.5) / 3.0
        dpdx = cs2 * self.delta_rho / (self.nx - 1)
        permeability = nu * u_mean / dpdx

        print(f"Step={timestep:7d} | BC={self.pressure_bc:10s} | <ux>={u_mean:.6e} | k={permeability:.6e}")

        fields = {
            "rho": rho[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
            "uz": u[..., 2],
            "solid": self.solid_mask.astype(np.int8),
        }
        save_fields_vtk(timestep, fields, "output", "porous_mrt_3d")


def parse_args():
    parser = argparse.ArgumentParser(description="3D MRT permeability with selectable pressure BC")
    parser.add_argument("--geometry", choices=["spheres", "cylinders"], default="spheres")
    parser.add_argument("--pressure-bc", choices=["zouhe", "regularized"], default="zouhe")
    parser.add_argument("--nx", type=int, default=120)
    parser.add_argument("--ny", type=int, default=72)
    parser.add_argument("--nz", type=int, default=72)
    parser.add_argument("--steps", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    precision = "f32/f32"
    omega = 1.3
    rho0 = 1.0
    delta_rho = 1e-3

    if args.geometry == "spheres":
        solid_mask = build_sphere_pack_mask(args.nx, args.ny, args.nz)
    else:
        solid_mask = build_cylinder_pack_mask(args.nx, args.ny, args.nz)

    porosity = 1.0 - np.mean(solid_mask)
    print(f"Geometry={args.geometry}, PressureBC={args.pressure_bc}, porosity={porosity:.4f}")

    os.system("rm -rf output* *.vtk")

    kwargs = {
        "lattice": LatticeD3Q19(precision),
        "omega": omega,
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "M": d3q19_mrt_matrix(),
        "s_rho": 0.0,
        "s_e": 1.1,
        "s_eta": 1.0,
        "s_j": 0.0,
        "s_q": 1.0,
        "s_v": omega,
        "s_m": 1.0,
        "s_pi": 1.0,
        "precision": precision,
        "io_rate": 1000,
        "print_info_rate": 1000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints"),
        "restore_checkpoint": False,
        "solid_mask": solid_mask,
        "rho0": rho0,
        "delta_rho": delta_rho,
        "pressure_bc": args.pressure_bc,
    }

    sim = PorousPermeabilityMRT3D(**kwargs)
    sim.run(args.steps)
