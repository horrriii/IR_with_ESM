"""A class for computing vibrational modes"""

import shutil
from math import pi, sqrt, log
import sys
from sys import stdout
import os
import json

import numpy as np
from pathlib import Path

import ase.units as units
import ase.io
from ase.parallel import world, paropen, parprint

from ase.utils.filecache import get_json_cache
from ase.io.espresso import grep_valence, Namelist
from ase.vibrations.data import VibrationsData

from collections import namedtuple


class AtomicDisplacements:
    def _disp(self, a, i, step):
        if isinstance(i, str):  # XXX Simplify by removing this.
            i = "xyz".index(i)
        return Displacement(a, i, np.sign(step), abs(step), self)

    def _eq_disp(self):
        return self._disp(0, 0, 0)

    @property
    def ndof(self):
        return 3 * len(self.indices)


class Displacement(namedtuple("Displacement", ["a", "i", "sign", "ndisp", "vib"])):
    @property
    def name(self):
        if self.sign == 0:
            return "eq"

        axisname = "xyz"[self.i]
        dispname = self.ndisp * " +-"[self.sign]
        return f"{self.a}{axisname}{dispname}"

    @property
    def _cached(self):
        return self.vib.cache[self.name]

    def forces(self):
        return self._cached["forces"].copy()

    @property
    def step(self):
        return self.ndisp * self.sign * self.vib.delta

    # XXX dipole only valid for infrared
    def dipole(self):
        return self._cached["dipole"].copy()

    # XXX below stuff only valid for TDDFT excitation stuff
    def save_ov_nn(self, ov_nn):
        np.save(Path(self.vib.exname) / (self.name + ".ov"), ov_nn)

    def load_ov_nn(self):
        return np.load(Path(self.vib.exname) / (self.name + ".ov.npy"))

    @property
    def _exname(self):
        return Path(self.vib.exname) / f"ex.{self.name}{self.vib.exext}"

    def calculate_and_save_static_polarizability(self, atoms):
        exobj = self.vib._new_exobj()
        excitation_data = exobj(atoms)
        np.savetxt(self._exname, excitation_data)

    def load_static_polarizability(self):
        return np.loadtxt(self._exname)

    def read_exobj(self):
        # XXX each exobj should allow for self._exname as Path
        return self.vib.read_exobj(str(self._exname))

    def calculate_and_save_exlist(self, atoms):
        # exo = self.vib._new_exobj()
        excalc = self.vib._new_exobj()
        exlist = excalc.calculate(atoms)
        # XXX each exobj should allow for self._exname as Path
        exlist.write(str(self._exname))


class Infrared_with_ESM(AtomicDisplacements):
    """Class for calculating vibrational modes and infrared intensities
    using finite difference.

    The vibrational modes are calculated from a finite difference
    approximation of the Hessian matrix.

    The *summary()*, *get_energies()* and *get_frequencies()* methods all take
    an optional *method* keyword.  Use method='Frederiksen' to use the method
    described in:

      T. Frederiksen, M. Paulsson, M. Brandbyge, A. P. Jauho:
      "Inelastic transport theory from first-principles: methodology and
      applications for nanoscale devices", Phys. Rev. B 75, 205413 (2007)

    atoms: Atoms object
        The atoms to work on.
    indices: list of int
        List of indices of atoms to vibrate.  Default behavior is
        to vibrate all atoms.
    name: str
        Name to use for files.
    delta: float
        Magnitude of displacements.
    nfree: int
        Number of displacements per atom and cartesian coordinate, 2 and 4 are
        supported. Default is 2 which will displace each atom +delta and
        -delta for each cartesian coordinate.
    directions : list of int
        Cartesian directions used to compute the dipole moment gradient.
        For example, `directions = [2]` considers only the dipole moment
        along the z-direction.
        This option should generally be `[2]` when using this module with ESM,
        so you usually do not need to modify it.
    calc_obj : calculator object
        The calculator used for computations (e.g., Espresso).
    calc_kwargs : dict
        Dictionary of parameters to pass to the calculator (e.g., input_data).
        **Important:**
        The `'pseudo_dir'` parameter **must** be specified, because the
        valence electrons of each atom are read from the pseudopotential file (`.p.p.`)
        in order to calculate the dipole moment from the `.esm1` file.

        Example:
        calc_kwargs = {
            "input_data": input_data,
            "profile": profile,  <----------------- EspressoProfile
            "pseudopotentials": pseudopotentials,
            "kpts": kpts,
            "directory": "DFT",
            "outdir": "tmp",
            "file_name": "prefix",
        }
    """

    def __init__(
        self,
        atoms,
        calc_obj,
        indices=None,
        name="ir",
        delta=0.01,
        nfree=2,
        directions=[2],
        calc_kwargs={},
    ):
        assert nfree in [2, 4]
        self.atoms = atoms
        self.calc = atoms.calc
        if indices is None:
            indices = range(len(atoms))
        if len(indices) != len(set(indices)):
            raise ValueError("one (or more) indices included more than once")
        self.indices = np.asarray(indices)
        self._name = name

        self.delta = delta
        self.nfree = nfree
        self.H = None
        self._vibrations = None
        self.__name__ = "Infrared_with_ESM"
        if directions is None:
            self.directions = np.asarray([0, 1, 2])
        else:
            self.directions = np.asarray(directions)

        # Calculator object and kwargs
        self.calc_obj = calc_obj
        self.calc_kwargs = calc_kwargs
        self.ir = True

        self.cache = get_json_cache(name)

    @property
    def name(self):
        return str(self.cache.directory)

    def run(self):
        """Run the vibration and Infrared calculations.

        This will calculate the forces for 6 displacements per atom +/-x,
        +/-y, +/-z. Only those calculations that are not already done will be
        started. Be aware that an interrupted calculation may produce an empty
        file (ending with .json), which must be deleted before restarting the
        job. Otherwise the forces will not be calculated for that
        displacement.

        This will save each .esm1 file at each displacement to a directory 'esm1'
        and calculate dipole moment from each .esm1 file.

        Note that the calculations for the different displacements can be done
        simultaneously by several independent processes. This feature relies
        on the existence of files and the subsequent creation of the file in
        case it is not found.

        If the program you want to use does not have a calculator in ASE, use
        ``iterdisplace`` to get all displaced structures and calculate the
        forces on your own.
        """

        if not self.cache.writable:
            raise RuntimeError(
                "Cannot run calculation.  "
                "Cache must be removed or split in order "
                "to have only one sort of data structure at a time."
            )

        self._check_old_pickles()

        now_dir = Path.cwd()
        json_dir = now_dir / self._name
        save_dir = now_dir / "output"
        os.makedirs(name=save_dir, exist_ok=True)

        pseudopotentials = self.calc_kwargs.get("pseudopotentials")
        outdir = self.calc_kwargs.get("outdir")
        directory_name = self.calc_kwargs.get("directory")
        file_name = self.calc_kwargs.get("file_name")

        for disp, atoms in self.iterdisplace(inplace=True):
            with self.cache.lock(disp.name) as handle:
                if handle is None:
                    continue

                result = self.calculate(atoms, disp)

                os.makedirs(save_dir / disp.name, exist_ok=True)
                if (
                    Namelist(self.calc_kwargs.get("input_data"))["control"]["trism"]
                    == True
                ):
                    esm1_src_file = (
                        now_dir / directory_name / outdir / f"{file_name}.esm1"
                    )
                    esm1_dst_file = save_dir / disp.name / f"{file_name}.esm1"

                    drism_src_file = (
                        now_dir / directory_name / outdir / f"{file_name}.1drism"
                    )
                    drism_dst_file = save_dir / disp.name / f"{file_name}.1drism"

                    rism_src_file = (
                        now_dir / directory_name / outdir / f"{file_name}.rism1"
                    )
                    rism_dst_file = save_dir / disp.name / f"{file_name}.rism1"

                    pwi_src_file = now_dir / directory_name / "espresso.pwi"
                    pwi_dst_file = save_dir / disp.name / "espresso.pwi"

                    pwo_src_file = now_dir / directory_name / "espresso.pwo"
                    pwo_dst_file = save_dir / disp.name / "espresso.pwo"

                    shutil.copy(src=esm1_src_file, dst=esm1_dst_file)
                    shutil.copy(src=drism_src_file, dst=drism_dst_file)
                    shutil.copy(src=rism_src_file, dst=rism_dst_file)
                    shutil.copy(src=pwi_src_file, dst=pwi_dst_file)
                    shutil.copy(src=pwo_src_file, dst=pwo_dst_file)
                else:
                    esm1_src_file = (
                        now_dir / directory_name / outdir / f"{file_name}.esm1"
                    )
                    esm1_dst_file = save_dir / disp.name / f"{file_name}.esm1"

                    pwi_src_file = now_dir / directory_name / "espresso.pwi"
                    pwi_dst_file = save_dir / disp.name / "espresso.pwi"

                    pwo_src_file = now_dir / directory_name / "espresso.pwo"
                    pwo_dst_file = save_dir / disp.name / "espresso.pwo"

                    shutil.copy(src=esm1_src_file, dst=esm1_dst_file)
                    shutil.copy(src=pwi_src_file, dst=pwi_dst_file)
                    shutil.copy(src=pwo_src_file, dst=pwo_dst_file)

                esm1 = np.loadtxt(esm1_dst_file, skiprows=1)

                pseudo_path = Namelist(self.calc_kwargs.get("input_data"))["control"][
                    "pseudo_dir"
                ]
                pseudo_path = Path(pseudo_path).expanduser()
                valence_electrons = []
                for i in range(len(atoms)):
                    valence_electrons.append(
                        grep_valence(pseudo_path / pseudopotentials[atoms.symbols[i]])
                    )

                valence_electrons = np.array(valence_electrons)
                mu_ion = atoms.positions[:, 2] * valence_electrons
                mu_ion = sum(mu_ion)
                z = esm1[:, 0]
                tot_chg = esm1[:, 1]
                integral = np.trapz(z * tot_chg, z)
                dipole = mu_ion - integral

                result_file = save_dir / "dipole.txt"
                with open(file=result_file, mode="a") as d:
                    d.write(f"{disp.name:<4} {dipole:>10.06f}\n")

                if world.rank == 0:
                    handle.save(result)

                filepath = os.path.join(json_dir, f"cache.{disp.name}.json")
                if not os.path.exists(filepath):
                    print(f"[WARNING] JSON file not found yet: {filepath}")
                else:
                    with open(filepath, "r") as f:
                        data = json.load(f)

                    data["dipole"] = {
                        "__ndarray__": [[3], "float64", [0.0, 0.0, dipole]]
                    }

                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

    def _check_old_pickles(self):
        from pathlib import Path

        eq_pickle_path = Path(f"{self.name}.eq.pckl")
        pickle2json_instructions = f"""\
Found old pickle files such as {eq_pickle_path}.  \
Please remove them and recalculate or run \
"python -m ase.vibrations.pickle2json --help"."""
        if len(self.cache) == 0 and eq_pickle_path.exists():
            raise RuntimeError(pickle2json_instructions)

    def iterdisplace(self, inplace=False):
        """Yield name and atoms object for initial and displaced structures.

        Use this to export the structures for each single-point calculation
        to an external program instead of using ``run()``. Then save the
        calculated gradients to <name>.json and continue using this instance.
        """
        # XXX change of type of disp
        atoms = self.atoms if inplace else self.atoms.copy()
        displacements = self.displacements()
        eq_disp = next(displacements)
        assert eq_disp.name == "eq"
        yield eq_disp, atoms

        for disp in displacements:
            if not inplace:
                atoms = self.atoms.copy()
            pos0 = atoms.positions[disp.a, disp.i]
            atoms.positions[disp.a, disp.i] += disp.step
            yield disp, atoms

            if inplace:
                atoms.positions[disp.a, disp.i] = pos0

    def iterimages(self):
        """Yield initial and displaced structures."""
        for name, atoms in self.iterdisplace():
            yield atoms

    def _iter_ai(self):
        for a in self.indices:
            for i in range(3):
                yield a, i

    def displacements(self):
        yield self._eq_disp()

        for a, i in self._iter_ai():
            for sign in [-1, 1]:
                for ndisp in range(1, self.nfree // 2 + 1):
                    yield self._disp(a, i, sign * ndisp)

    def calculate(self, atoms, disp):
        # Set calculator
        calc_kwargs = self.calc_kwargs.copy()
        calc_kwargs = self.config_allowforce(calc_kwargs)  # Config allowforce
        calc = self.calc_obj(**calc_kwargs)

        results = {}
        results["forces"] = calc.get_forces(atoms)

        # if self.ir:
        #     results["dipole"] = self.calc.get_dipole_moment(atoms)

        return results

    def clean(self, empty_files=False, combined=True):
        """Remove json-files.

        Use empty_files=True to remove only empty files and
        combined=False to not remove the combined file.

        """

        if world.rank != 0:
            return 0

        if empty_files:
            return self.cache.strip_empties()  # XXX Fails on combined cache

        nfiles = self.cache.filecount()
        self.cache.clear()
        return nfiles

    def combine(self):
        """Combine json-files to one file ending with '.all.json'.

        The other json-files will be removed in order to have only one sort
        of data structure at a time.

        """
        nelements_before = self.cache.filecount()
        self.cache = self.cache.combine()
        return nelements_before

    def split(self):
        """Split combined json-file.

        The combined json-file will be removed in order to have only one
        sort of data structure at a time.

        """
        count = self.cache.filecount()
        self.cache = self.cache.split()
        return count

    def read(self, method="standard", direction="central"):
        self.method = method.lower()
        self.direction = direction.lower()
        assert self.method in ["standard", "frederiksen"]

        if direction != "central":
            raise NotImplementedError(
                "Only central difference is implemented at the moment."
            )

        disp = self._eq_disp()
        forces_zero = disp.forces()
        dipole_zero = disp.dipole()
        self.dipole_zero = (sum(dipole_zero**2) ** 0.5) / units.Debye
        self.force_zero = max(sum((forces_zero[j]) ** 2) ** 0.5 for j in self.indices)

        ndof = 3 * len(self.indices)
        H = np.empty((ndof, ndof))
        dpdx = np.empty((ndof, 3))
        for r, (a, i) in enumerate(self._iter_ai()):
            disp_minus = self._disp(a, i, -1)
            disp_plus = self._disp(a, i, 1)

            fminus = disp_minus.forces()
            dminus = disp_minus.dipole()

            fplus = disp_plus.forces()
            dplus = disp_plus.dipole()

            if self.nfree == 4:
                disp_mm = self._disp(a, i, -2)
                disp_pp = self._disp(a, i, 2)
                fminusminus = disp_mm.forces()
                dminusminus = disp_mm.dipole()

                fplusplus = disp_pp.forces()
                dplusplus = disp_pp.dipole()
            if self.method == "frederiksen":
                fminus[a] += -fminus.sum(0)
                fplus[a] += -fplus.sum(0)
                if self.nfree == 4:
                    fminusminus[a] += -fminus.sum(0)
                    fplusplus[a] += -fplus.sum(0)
            if self.nfree == 2:
                H[r] = (fminus - fplus)[self.indices].ravel() / 2.0
                dpdx[r] = dminus - dplus
            if self.nfree == 4:
                H[r] = (-fminusminus + 8 * fminus - 8 * fplus + fplusplus)[
                    self.indices
                ].ravel() / 12.0
                dpdx[r] = (-dplusplus + 8 * dplus - 8 * dminus + dminusminus) / 6.0
            H[r] /= 2 * self.delta
            dpdx[r] /= 2 * self.delta
            for n in range(3):
                if n not in self.directions:
                    dpdx[r][n] = 0
                    dpdx[r][n] = 0
        # Calculate eigenfrequencies and eigenvectors
        masses = self.atoms.get_masses()
        H += H.copy().T
        self.H = H

        self.im = np.repeat(masses[self.indices] ** -0.5, 3)
        omega2, modes = np.linalg.eigh(self.im[:, None] * H * self.im)
        self.modes = modes.T.copy()

        # Calculate intensities
        dpdq = np.array(
            [
                dpdx[j] / sqrt(masses[self.indices[j // 3]] * units._amu / units._me)
                for j in range(ndof)
            ]
        )
        dpdQ = np.dot(dpdq.T, modes)
        dpdQ = dpdQ.T
        intensities = np.array([sum(dpdQ[j] ** 2) for j in range(ndof)])
        # Conversion factor:
        s = units._hbar * 1e10 / sqrt(units._e * units._amu)
        self.hnu = s * omega2.astype(complex) ** 0.5
        # Conversion factor from atomic units to (D/Angstrom)^2/amu.
        conv = (1.0 / units.Debye) ** 2 * units._amu / units._me
        self.intensities = intensities * conv

    def intensity_prefactor(self, intensity_unit):
        if intensity_unit == "(D/A)2/amu":
            return 1.0, "(D/Å)^2 amu^-1"
        elif intensity_unit == "km/mol":
            # conversion factor from Porezag PRB 54 (1996) 7830
            return 42.255, "km/mol"
        else:
            raise RuntimeError("Intensity unit >" + intensity_unit + "< unknown.")

    def get_vibrations(
        self, method="standard", direction="central", read_cache=True, **kw
    ):
        """Get vibrations as VibrationsData object

        If read() has not yet been called, this will be called to assemble data
        from the outputs of run(). Most of the arguments to this function are
        options to be passed to read() in this case.

        Args:
            method (str): Calculation method passed to read()
            direction (str): Finite-difference scheme passed to read()
            read_cache (bool): The VibrationsData object will be cached for
                quick access. Set False to force regeneration of the cache with
                the current atoms/Hessian/indices data.
            **kw: Any remaining keyword arguments are passed to read()

        Returns:
            VibrationsData

        """
        if read_cache and (self._vibrations is not None):
            return self._vibrations

        else:
            if (
                self.H is None
                or method.lower() != self.method
                or direction.lower() != self.direction
            ):
                self.read(method, direction, **kw)

            return VibrationsData.from_2d(self.atoms, self.H, indices=self.indices)

    def get_energies(self, method="standard", direction="central", **kw):
        """Get vibration energies in eV."""
        return self.get_vibrations(
            method=method, direction=direction, **kw
        ).get_energies()

    def get_frequencies(self, method="standard", direction="central"):
        """Get vibration frequencies in cm^-1."""
        return self.get_vibrations(method=method, direction=direction).get_frequencies()

    def summary(
        self,
        method="standard",
        direction="central",
        intensity_unit="(D/A)2/amu",
        log=stdout,
    ):
        hnu = self.get_energies(method, direction)
        s = 0.01 * units._e / units._c / units._hplanck
        iu, iu_string = self.intensity_prefactor(intensity_unit)
        if intensity_unit == "(D/A)2/amu":
            iu_format = "%9.4f"
        elif intensity_unit == "km/mol":
            iu_string = "   " + iu_string
            iu_format = " %7.1f"
        if isinstance(log, str):
            log = paropen(log, "a")

        parprint("-------------------------------------", file=log)
        parprint(" Mode    Frequency        Intensity", file=log)
        parprint("  #    meV     cm^-1   " + iu_string, file=log)
        parprint("-------------------------------------", file=log)
        for n, e in enumerate(hnu):
            if e.imag != 0:
                c = "i"
                e = e.imag
            else:
                c = " "
                e = e.real
            parprint(
                ("%3d %6.1f%s  %7.1f%s  " + iu_format)
                % (n, 1000 * e, c, s * e, c, iu * self.intensities[n]),
                file=log,
            )
        parprint("-------------------------------------", file=log)
        parprint("Zero-point energy: %.3f eV" % self.get_zero_point_energy(), file=log)
        parprint("Static dipole moment: %.3f D" % self.dipole_zero, file=log)
        parprint(
            "Maximum force on atom in `equilibrium`: %.4f eV/Å" % self.force_zero,
            file=log,
        )
        parprint(file=log)

    def get_zero_point_energy(self, freq=None):
        if freq:
            raise NotImplementedError()
        return self.get_vibrations().get_zero_point_energy()

    def get_mode(self, n):
        """Get mode number ."""
        return self.get_vibrations().get_modes(all_atoms=True)[n]

    def write_mode(self, n=None, kT=units.kB * 300, nimages=30):
        """Write mode number n to trajectory file. If n is not specified,
        writes all non-zero modes."""
        if n is None:
            for index, energy in enumerate(self.get_energies()):
                if abs(energy) > 1e-5:
                    self.write_mode(n=index, kT=kT, nimages=nimages)
            return

        else:
            n %= len(self.get_energies())

        with ase.io.Trajectory("%s.%d.traj" % (self.name, n), "w") as traj:
            for image in self.get_vibrations().iter_animated_mode(
                n, temperature=kT, frames=nimages
            ):
                traj.write(image)

    def show_as_force(self, n, scale=0.2, show=True):
        return self.get_vibrations().show_as_force(n, scale=scale, show=show)

    def write_jmol(self):
        """Writes file for viewing of the modes with jmol."""

        with open(self.name + ".xyz", "w") as fd:
            self._write_jmol(fd)

    def _write_jmol(self, fd):
        symbols = self.atoms.get_chemical_symbols()
        freq = self.get_frequencies()
        for n in range(3 * len(self.indices)):
            fd.write("%6d\n" % len(self.atoms))

            if freq[n].imag != 0:
                c = "i"
                freq[n] = freq[n].imag

            else:
                freq[n] = freq[n].real
                c = " "

            fd.write("Mode #%d, f = %.1f%s cm^-1" % (n, float(freq[n].real), c))

            if self.ir:
                fd.write(", I = %.4f (D/Å)^2 amu^-1.\n" % self.intensities[n])
            else:
                fd.write(".\n")

            mode = self.get_mode(n)
            for i, pos in enumerate(self.atoms.positions):
                fd.write(
                    "%2s %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f\n"
                    % (
                        symbols[i],
                        pos[0],
                        pos[1],
                        pos[2],
                        mode[i, 0],
                        mode[i, 1],
                        mode[i, 2],
                    )
                )

    def fold(
        self,
        frequencies,
        intensities,
        start=800.0,
        end=4000.0,
        npts=None,
        width=4.0,
        type="Gaussian",
        normalize=False,
    ):
        """Fold frequencies and intensities within the given range
        and folding method (Gaussian/Lorentzian).
        The energy unit is cm^-1.
        normalize=True ensures the integral over the peaks to give the
        intensity.
        """

        lctype = type.lower()
        assert lctype in ["gaussian", "lorentzian"]
        if not npts:
            npts = int((end - start) / width * 10 + 1)
        prefactor = 1
        if lctype == "lorentzian":
            intensities = intensities * width * pi / 2.0
            if normalize:
                prefactor = 2.0 / width / pi
        else:
            sigma = width / 2.0 / sqrt(2.0 * log(2.0))
            if normalize:
                prefactor = 1.0 / sigma / sqrt(2 * pi)

        # Make array with spectrum data
        spectrum = np.empty(npts)
        energies = np.linspace(start, end, npts)
        for i, energy in enumerate(energies):
            energies[i] = energy
            if lctype == "lorentzian":
                spectrum[i] = (
                    intensities
                    * 0.5
                    * width
                    / pi
                    / ((frequencies - energy) ** 2 + 0.25 * width**2)
                ).sum()
            else:
                spectrum[i] = (
                    intensities
                    * np.exp(-((frequencies - energy) ** 2) / 2.0 / sigma**2)
                ).sum()
        return [energies, prefactor * spectrum]

    def write_dos(
        self,
        out="vib-dos.dat",
        start=800,
        end=4000,
        npts=None,
        width=10,
        type="Gaussian",
        method="standard",
        direction="central",
    ):
        """Write out the vibrational density of states to file.

        First column is the wavenumber in cm^-1, the second column the
        folded vibrational density of states.
        Start and end points, and width of the Gaussian/Lorentzian
        should be given in cm^-1."""
        frequencies = self.get_frequencies(method, direction).real
        intensities = np.ones(len(frequencies))
        energies, spectrum = self.fold(
            frequencies, intensities, start, end, npts, width, type
        )

        # Write out spectrum in file.
        outdata = np.empty([len(energies), 2])
        outdata.T[0] = energies
        outdata.T[1] = spectrum

        with open(out, "w") as fd:
            fd.write("# %s folded, width=%g cm^-1\n" % (type.title(), width))
            fd.write("# [cm^-1] arbitrary\n")
            for row in outdata:
                fd.write("%.3f  %15.5e\n" % (row[0], row[1]))

    def write_spectra(
        self,
        out="ir-spectra.dat",
        start=800,
        end=4000,
        npts=None,
        width=10,
        type="Gaussian",
        method="standard",
        direction="central",
        intensity_unit="(D/A)2/amu",
        normalize=False,
    ):
        """Write out infrared spectrum to file.

        First column is the wavenumber in cm^-1, the second column the
        absolute infrared intensities, and
        the third column the absorbance scaled so that data runs
        from 1 to 0. Start and end
        point, and width of the Gaussian/Lorentzian should be given
        in cm^-1."""
        energies, spectrum = self.get_spectrum(
            start, end, npts, width, type, method, direction, normalize
        )

        # Write out spectrum in file. First column is absolute intensities.
        # Second column is absorbance scaled so that data runs from 1 to 0
        spectrum2 = 1.0 - spectrum / spectrum.max()
        outdata = np.empty([len(energies), 3])
        outdata.T[0] = energies
        outdata.T[1] = spectrum
        outdata.T[2] = spectrum2
        with open(out, "w") as fd:
            fd.write(f"# {type.title()} folded, width={width:g} cm^-1\n")
            iu, iu_string = self.intensity_prefactor(intensity_unit)
            if normalize:
                iu_string = "cm " + iu_string
            fd.write("# [cm^-1] %14s\n" % ("[" + iu_string + "]"))
            for row in outdata:
                fd.write("%.3f  %15.5e  %15.5e \n" % (row[0], iu * row[1], row[2]))

    def get_spectrum(
        self,
        start=800,
        end=4000,
        npts=None,
        width=4,
        type="Gaussian",
        method="standard",
        direction="central",
        intensity_unit="(D/A)2/amu",
        normalize=False,
    ):
        """Get infrared spectrum.

        The method returns wavenumbers in cm^-1 with corresponding
        absolute infrared intensity.
        Start and end point, and width of the Gaussian/Lorentzian should
        be given in cm^-1.
        normalize=True ensures the integral over the peaks to give the
        intensity.
        """
        frequencies = self.get_frequencies(method, direction).real
        intensities = self.intensities
        return self.fold(
            frequencies, intensities, start, end, npts, width, type, normalize
        )

    def config_allowforce(self, kwargs):
        kwargs = kwargs.copy()

        # Espresso-case
        if self.calc_obj.__name__ == "Espresso":
            input_data = Namelist(kwargs.pop("input_data", None))
            input_data["control"].setdefault("tprnfor", True)
            input_data["control"].setdefault(
                "pseudo_dir", str(Path("~/QE/pseudo/").expanduser())
            )
            input_data["control"].setdefault("trism", False)
            input_data["control"]["prefix"] = kwargs.get("file_name")
            input_data["control"]["outdir"] = kwargs.get("outdir")
            kwargs["input_data"] = input_data

        return kwargs
