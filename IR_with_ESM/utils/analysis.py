from pathlib import Path
from IR_with_ESM.utils.plot import plot_atoms, plot_cell, draw_axes
from IR_with_ESM import Infrared_with_ESM
from typing import Union
from ase.io import read, write
import matplotlib.pyplot as plt
import shutil
import numpy as np
import pandas as pd
import imageio
from tqdm.contrib.concurrent import process_map


class Analysis:
    def __init__(
        self,
        vibir_obj: Infrared_with_ESM,
        cache: Union[Path, str] = "vibir_analysis",
    ):
        self.vibir_obj = vibir_obj
        self.vibir_data = vibir_obj.get_vibrations()
        self.vibir_name = vibir_obj.__name__

        self.n = len(self.vibir_data.get_modes())
        self.nchar = len(str(self.n))

        self.cache = Path(cache)

    # def get_summary(self, im_tol: float = 1e-08, log=None):
    def get_summary(self, **kwargs):
        self.vibir_obj.summary(**kwargs)

    def export_gif(self, parallel: int = 1, frames: int = 15):
        """
        Export GIFs of each mode

        Args:
            parallel (int, optional): Number of parallel processes. Defaults to 1.
            frames (int, optional): Number of frames per mode. Defaults to 15.

        Returns:
            list: List of results
        """

        loc = self.cache / f"{self.vibir_name}_mode_gif"
        loc.mkdir(exist_ok=True, parents=True)
        traj_dir = self.cache / f"{self.vibir_name}_mode_traj"
        traj_dir.mkdir(exist_ok=True, parents=True)
        for i in range(self.n):
            shutil.move(f"ir.{i}.traj", traj_dir / f"mode_{i}.traj")

        def get_animated_modes(i):
            return list(self.vibir_data.iter_animated_mode(i, frames=frames))

        args = [
            (
                loc,
                get_animated_modes(i),
                i,
                self.nchar,
            )
            for i in range(self.n)
        ]

        results = process_map(
            write_gif,
            args,
            max_workers=parallel,
            desc="Exporting GIFs",
        )

        return results

    def get_spectra(
        self,
        start=0,
        end=5000,
        width=30,
        npts=None,
        type="Lorentzian",
        normalize=False,
        method="standard",
        direction="central",
    ):

        frequencies = self.vibir_obj.get_frequencies(
            method=method, direction=direction
        ).real

        if "Vibrations" in self.vibir_name:
            intensities = np.ones_like(frequencies)

        if "Infrared" in self.vibir_name:
            intensities = self.vibir_obj.intensities

        energies, spectrum = self.vibir_obj.fold(
            frequencies,
            intensities,
            start=start,
            end=end,
            npts=npts,
            width=width,
            type=type,
            normalize=normalize,
        )

        if "Vibrations" in self.vibir_name:
            spectra = np.stack((energies, spectrum), axis=1)
            return frequencies, spectra

        if "Infrared" in self.vibir_name:
            spectrum2 = 1.0 - spectrum / spectrum.max()
            spectra = np.stack((energies, spectrum, spectrum2), axis=1)

        return frequencies, spectra

    def get_spectra_plot(self, ax=None, save=None, **kwargs):
        frequencies, spectra = self.get_spectra(**kwargs)

        if "Vibrations" in self.vibir_name:
            name = "vib"
        if "Infrared" in self.vibir_name:
            name = "ir"

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        # Plot spectra
        ax.fill_between(spectra[:, 0], spectra[:, 1], color="k", alpha=0.8, lw=0.5)

        # Plot dicrete mode location

        if name == "vib":
            ymax_data = []
            for i in frequencies:
                index = np.argmin(np.abs(spectra[:, 0] - i))
                ymax_data.append(spectra[index, 1])
            ymax_data = np.array(ymax_data, dtype=float)

        if name == "ir":
            ymax_data = self.vibir_obj.intensities

        color = "limegreen" if name == "vib" else "red"
        ax.vlines(frequencies, 0, ymax_data, color=color, lw=1)

        # Set labels
        ax.set_xlabel("Frequency, cm$^{-1}$", fontsize=12)
        if name == "vib":
            ax.set_ylabel("VDOS", fontsize=12)
        if name == "ir":
            ax.set_ylabel("Intensity, (D/Å)$^2$ amu$^{-1}$", fontsize=12)

        # Save file protocol
        if save and fig is not None:
            fig.savefig(save, dpi=300, bbox_inches="tight")

        if fig is None:
            return ax
        else:
            return fig, ax


def write_traj(args: tuple):
    """
    Write trajectory from list of atoms

    Args:
        args (tuple): Tuple containing loc, atoms, i, nchar

    Returns:
        int: 01"""
    loc, atoms, i, nchar = args
    write(loc / f"mode_{i:0{nchar}}.traj", atoms)
    return 0


def write_gif(args: tuple):
    """
    Write gif from list of atoms

    Args:
        args (tuple): Tuple containing loc, atoms, i, nchar

    Returns:
        int: 0
    """

    loc, atoms, i, nchar = args

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    fnames = []

    # Make images
    for j, atom in enumerate(atoms):
        tmp = loc / f"tmp_mode_{i:0{nchar}}"
        tmp.mkdir(exist_ok=True, parents=True)

        name = tmp / f"mode_{i:0{nchar}}_{j:03}.png"
        fnames.append(name)

        for ax, view in zip(axs, ["xy+", "xz+", "yz+"]):
            ax.cla()
            plot_atoms(
                ax=ax,
                atoms=atom,
                plot_constraint=True,
                plane=view,
            )

            plot_cell(ax=ax, cell=atom.get_cell(), plane=view)
            draw_axes(ax, view, length=2.0)

            if view == "yz+":
                ax.set_ylim(0 - 1, atom.get_cell().sum(axis=0)[2] + 1)
                ax.set_xlim(0 - 1, atom.get_cell().sum(axis=0)[1] + 1)
            if view == "xz+":
                ax.set_xlim(0 - 1, atom.get_cell().sum(axis=0)[0] + 1)
                ax.set_ylim(0 - 1, atom.get_cell().sum(axis=0)[2] + 1)
            if view == "xy+":
                ax.set_xlim(0 - 1, atom.get_cell().sum(axis=0)[0] + 1)
                ax.set_ylim(0 - 1, atom.get_cell().sum(axis=0)[1] + 1)

            ax.set_aspect("equal")
            ax.axis("off")

        fig.savefig(str(name), dpi=300)
        plt.close(fig)

    # Compile images into gif
    gif_name = loc / f"mode_{i:0{nchar}}.gif"

    images = [imageio.imread(name) for name in fnames]
    fps = 15

    imageio.mimsave(gif_name, images, loop=0, fps=fps)

    shutil.rmtree(tmp)

    return 0