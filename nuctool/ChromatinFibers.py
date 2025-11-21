"""
ChromatinFibers: A comprehensive toolkit for chromatin fiber simulation and analysis.

This module provides tools for simulating and analyzing chromatin fiber structures,
nucleosome positioning, and DNA methylation patterns. It implements physical models
of DNA-histone interactions including wrapping energy calculations, steric exclusion
effects, and thermodynamic equilibrium sampling.

Key Features:
    - DNA sequence-dependent nucleosome positioning prediction
    - Wrapping energy calculations based on dinucleotide periodicity
    - Statistical mechanics modeling (Vanderlick formalism)
    - Monte Carlo sampling of nucleosome configurations
    - DNA methylation pattern simulation
    - Footprint analysis from methylation data
    - Integration with genomic databases and annotation files
    - HDF5-based batch simulation framework

Main Classes:
    - ChromatinFiber: Core class for chromatin fiber modeling and simulation
    - SimulationParams: Configuration dataclass for batch simulations
    - WrappingEnergyResult: Container for wrapping energy components
    - MethylationResult: Container for methylation and protection data

Main Functions:
    - simulate_chromatin_fibers: Batch simulation with HDF5 output
    - read_simulation_results: Load and parse simulation results
    - compute_vanderlick: Statistical mechanics equilibrium solver
    - convert_to_footprints: Extract footprints from methylation patterns

Example:
    >>> fiber = ChromatinFiber(sequence="ATCG"*1000, start=0)
    >>> fiber.calc_energy_landscape(amplitude=0.05, period=10.0)
    >>> dyads = fiber.sample_fiber_configuration()
    >>> methylation = fiber.calc_methylation(dyads, efficiency=0.7)

Author: J. van Noort
Date: 2025
"""

from dataclasses import dataclass
from fileinput import filename
import numpy as np
import pandas as pd
import re
from pathlib import Path
from snapgene_reader import snapgene_file_to_dict
import matplotlib.pyplot as plt
from icecream import ic
from Bio import Entrez, SeqIO
import matplotlib.pyplot as plt
from tqdm import tqdm
import genomepy
import h5py
from pathlib import Path
from .Plotter import Plotter, FIGSIZE


FOOTPRINT = 146  # Nucleosome DNA footprint size in base pairs

# Optimized lookup table for DNA sequence encoding
# Maps ASCII characters to numeric indices for vectorized operations
_ASCII_TO_IDX = np.full(256, -1, dtype=np.int8)
_ASCII_TO_IDX[ord("A")] = 0
_ASCII_TO_IDX[ord("C")] = 1
_ASCII_TO_IDX[ord("G")] = 2
_ASCII_TO_IDX[ord("T")] = 3
_ASCII_TO_IDX[ord("a")] = 4
_ASCII_TO_IDX[ord("c")] = 5
_ASCII_TO_IDX[ord("g")] = 6
_ASCII_TO_IDX[ord("t")] = 7


@dataclass(frozen=True)
class WrappingEnergyResult:
    """Container for nucleosome wrapping energy components.

    Attributes:
        octamer: Total wrapping energy for full nucleosome (octamer).
        tetramer: Partial wrapping energy for tetrasome core.
        segments: Array of wrapping energies for 14 DNA contact segments.
    """

    octamer: float
    tetramer: float
    segments: np.ndarray


@dataclass(frozen=True)
class MethylationResult:
    """Container for DNA methylation simulation results.

    Attributes:
        protected: Boolean array indicating nucleosome-protected positions.
        methylated: Boolean array indicating successfully methylated positions.
    """

    protected: np.ndarray
    methylated: np.ndarray


@dataclass
class SimulationParams:
    """Parameters for chromatin fiber simulation.

    Attributes:
        n_samples: Number of fiber configurations to sample.
        length_bp: Length of DNA sequence in base pairs.
        amplitude: Amplitude of periodic dinucleotide preference (0-1).
        period_bp: Period of DNA bending preference in base pairs (~10 bp).
        chemical_potential_kT: Nucleosome binding chemical potential in kT units.
        e_contact_kT: Energy per DNA-histone contact point in kT units.
        motifs: Tuple of DNA motifs to search for in methylation analysis.
        strand: Strand specification for motif search ("both", "plus", "minus").
        efficiency: Methylation efficiency (0-1).
        steric_exclusion_bp: Excluded footprint size for nucleosome sampling.
    """

    n_samples: int = 1000
    length_bp: int = 10_000
    amplitude: float = 0.05
    period_bp: float = 10.0
    chemical_potential_kT: float = 0.0
    e_contact_kT: float = -0.5
    motifs: list = ("A",)
    strand: str = "both"
    efficiency: float = 0.7
    steric_exclusion_bp: int = 0


def convert_to_footprints(methylated, index, minimal_footprint=10):
    """Convert methylation patterns to nucleosome footprints.

    Identifies protected regions (footprints) from methylation accessibility data
    by finding gaps between methylated positions.

    Args:
        methylated: List of boolean arrays indicating methylation at each position.
        index: Array of genomic positions corresponding to methylation data.
        minimal_footprint: Minimum footprint size in bp to include (default 10).

    Returns:
        DataFrame with columns: read_id, start, end, width.

    Example:
        >>> footprints = convert_to_footprints(methylated, positions, minimal_footprint=100)
        >>> avg_size = footprints['width'].mean()
    """
    footprints = []
    idx = np.asarray(index)

    for read_id, trace in enumerate(methylated):
        methylations = idx[np.flatnonzero(np.asarray(trace) == 1)]

        lengths = np.diff(methylations)

        starts = methylations[:-1]
        ends = methylations[1:]

        for start, end in zip(starts, ends):
            footprints.append([int(read_id), int(start), int(end), int(end - start)])

    if footprints:
        df = pd.DataFrame.from_records(
            footprints, columns=["read_id", "start", "end", "width"]
        )
    else:
        df = pd.DataFrame(columns=["read_id", "start", "end", "width"])

    return df[df["width"] >= minimal_footprint]


def encode_seq(seq: str) -> np.ndarray:
    """Convert DNA sequence string to numeric indices for vectorized computation.

    Maps nucleotide characters to integers: A→0, C→1, G→2, T→3 (uppercase),
    and a→4, c→5, g→6, t→7 (lowercase). Other characters map to -1.

    Args:
        seq: DNA sequence string (can be str or Biopython Seq object).

    Returns:
        NumPy array of int8 indices for each nucleotide.

    Example:
        >>> encode_seq("ATCG")
        array([0, 3, 1, 2], dtype=int8)
    """
    # Convert Biopython Seq object to string if needed
    seq_str = str(seq)
    b = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
    return _ASCII_TO_IDX[b]


def get_weight(
    w: int, period: float, amplitude: float, show: bool = False
) -> np.ndarray:
    """Generate periodic dinucleotide weights to model DNA bendability.

    Creates position-dependent dinucleotide probability weights that capture
    the ~10 bp periodic preference for A/T-rich dinucleotides where DNA bends
    toward the histone octamer.

    The model implements:
    - AA, TA dinucleotides preferred where DNA bends inward (positive phase)
    - GC dinucleotides anti-preferred at the same positions
    - Periodic modulation with specified amplitude and period

    Args:
        w: Window size (typically 146 bp for nucleosomes).
        period: Periodicity of DNA bending preference in bp (typically ~10 bp).
        amplitude: Modulation amplitude (0-0.25, typically ~0.05).
        show: If True, plot the dinucleotide weight patterns.

    Returns:
        4D array of shape (4, 4, w) with probability weights for each
        dinucleotide at each position. Indices correspond to [base1, base2, position].

    Example:
        >>> weights = get_weight(146, period=10.0, amplitude=0.05)
        >>> log_weights = np.log(weights)
    """
    x = np.arange(w, dtype=np.int32) - w // 2
    s = amplitude * np.cos(2 * np.pi * x / period)
    weight = np.empty((4, 4, w), dtype=np.float64)
    weight[0, 0] = 0.25 + s
    weight[0, 1] = 0.25 - s / 3
    weight[0, 2] = 0.25 - s / 3
    weight[0, 3] = 0.25 - s / 3
    weight[1, 0] = 0.25
    weight[1, 1] = 0.25
    weight[1, 2] = 0.25
    weight[1, 3] = 0.25
    weight[2, 0] = 0.25 + s / 3
    weight[2, 1] = 0.25 - s
    weight[2, 2] = 0.25 + s / 3
    weight[2, 3] = 0.25 + s / 3
    weight[3, 0] = 0.25 + s
    weight[3, 1] = 0.25 - s
    weight[3, 2] = 0.25 - s
    weight[3, 3] = 0.25 + s

    if show:
        fig, axes = plt.subplots(2, 2, figsize=(FIGSIZE[0], FIGSIZE[1]))
        axes = axes.flatten()

        for idx, base1 in enumerate("ACTG"):
            ax = axes[idx]
            label2 = ""
            base2 = "A"
            w1 = weight[*encode_seq(base1 + base2)]
            w2 = -np.ones_like(w1)
            label1 = base1 + base2
            for base2 in "CGT":
                width = weight[*encode_seq(base1 + base2)]
                if np.array_equal(width, w1):
                    label1 += ", " + base1 + base2
                else:
                    w2 = width
                    if label2 is None:
                        label2 = base1 + base2
                    else:
                        if len(label2) > 0:
                            label2 += ", "
                        label2 += base1 + base2
            if len(label1) > len(label2):
                label1, label2 = label2, label1
                w1, w2 = w2, w1

            if w1 is not None:
                ax.plot(x, w1, color="red", label=label1)
            if w2 is not None:
                ax.plot(x, w2, color="black", label=label2)

            ax.axvline(0, color="grey", linestyle="dashed", alpha=0.5)
            ax.legend(loc="upper right", framealpha=0.9, facecolor="white")
            ax.set_xlabel("i (bp)")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 0.5)
            ax.set_xlim(-FOOTPRINT // 2, FOOTPRINT // 2)
            plt.tight_layout()

    return weight


def calc_wrapping_energy(
    sequence: str, dyad: int, log_weights: np.ndarray, show: str | None = None
) -> WrappingEnergyResult:
    """Calculate DNA wrapping energy for a nucleosome at a specific dyad position.

    Computes the sequence-dependent wrapping energy by summing log-probability
    weights for all dinucleotides in the wrapped DNA segment. The energy is
    calculated symmetrically for both DNA strands (nucleosome has 2-fold symmetry).

    The calculation also segments the energy into 14 contact regions between
    DNA and histones for unwrapping thermodynamics analysis.

    Args:
        sequence: DNA sequence string.
        dyad: Central dyad position (0-indexed).
        log_weights: Log-probability weights from get_weight() [shape: 4,4,w].
        show: If specified, show diagnostic plots (not implemented).

    Returns:
        WrappingEnergyResult with:
            - octamer: Total energy for full nucleosome wrapping
            - tetramer: Energy for central tetrasome region
            - segments: Array of 14 segment energies for unwrapping analysis

    Note:
        Returns NaN values if dyad position is too close to sequence boundaries.

    Example:
        >>> weights = get_weight(146, period=10.0, amplitude=0.05)
        >>> result = calc_wrapping_energy(sequence, dyad=500, np.log(weights))
        >>> print(f"Wrapping energy: {result.octamer:.2f} kT")
    """
    start = -len(log_weights[0, 0]) // 2
    end = start + len(log_weights[0, 0]) + 1
    if dyad + start < 0 or dyad + end - 1 >= len(sequence):
        return WrappingEnergyResult(
            octamer=np.nan, tetramer=np.nan, segments=np.full(14, np.nan)
        )

    # Extract and encode sequence segment
    seq_segment = sequence[dyad + start : dyad + end - 1]
    idx = encode_seq(seq_segment.upper())

    # sum log-weights for both orientations as nucleosome is symmetric
    n = len(idx) - 1
    indices = np.arange(n)
    forward = log_weights[idx[:-1], idx[1:], indices] + np.log(4.0)
    forward = np.append(forward, np.nan)
    idx_rev = idx[::-1]
    reverse = log_weights[idx_rev[:-1], idx_rev[1:], indices] + np.log(4.0)
    reverse = np.append(reverse, np.nan)[::-1]
    energy = forward + reverse

    # add up energies of segments between contact points
    # make sure that segment boundaries are included for unwrapping from the outside of the octamers
    contacts = np.arange(-7, 7).astype(np.int16) * 10 + 5
    x_positions = np.arange(start, end - 1).astype(np.float64)

    tetramer = np.where((x_positions > -36) & (x_positions < 36))
    octamer = np.where((x_positions > -66) & (x_positions < 66))

    wrapping_energy = np.asarray(
        [
            np.sum(
                energy[(x_positions >= contacts[i]) & (x_positions < contacts[i + 1])]
            )
            for i in range(0, 6)
        ]
        + [np.sum(energy[(x_positions >= contacts[6]) & (x_positions <= contacts[7])])]
        + [
            np.sum(
                energy[(x_positions > contacts[i]) & (x_positions <= contacts[i + 1])]
            )
            for i in range(7, 13)
        ]
    )
    octamer_energy = np.sum(energy[octamer])
    tetramer_energy = np.sum(energy[tetramer])
    return WrappingEnergyResult(
        octamer=octamer_energy, tetramer=tetramer_energy, segments=wrapping_energy
    )


def compute_vanderlick(
    wrapping_energy: np.ndarray, show: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Compute equilibrium dyad probability using Vanderlick statistical mechanics formalism.

    Solves the statistical mechanical partition function for nucleosome positioning
    with steric exclusion, accounting for the fact that nucleosomes cannot overlap
    due to their 146 bp footprint. Uses forward-backward dynamic programming.

    This implementation follows the Vanderlick formalism for hard-core interactions
    on a lattice, providing the exact thermodynamic equilibrium distribution of
    nucleosomes given the position-dependent wrapping energies.

    Args:
        wrapping_energy: Array of wrapping energies for each position (in kT units).
        show: If True, plot the resulting dyad probabilities and occupancy.

    Returns:
        Tuple of (dyads, occupancy):
            - dyads: Probability density of nucleosome dyad at each position
            - occupancy: Total nucleosome occupancy (summed over 146 bp footprint)

    Note:
        Positions with NaN wrapping energy are excluded (probability set to 0).
        Occupancy is clipped to [0, 1] to handle numerical precision issues.

    Example:
        >>> energy = fiber.calc_energy_landscape(amplitude=0.05, period=10.0)
        >>> dyads, occupancy = compute_vanderlick(energy)
        >>> plt.plot(occupancy)

    Reference:
        Adapted from Vanderlick et al. statistical mechanics of hard rods on a line.
    """
    footprint = FOOTPRINT
    free_energy = wrapping_energy
    free_energy = np.nan_to_num(free_energy, nan=np.nanmax(free_energy))
    free_energy *= -1

    n = free_energy.size
    forward = np.zeros(n, dtype=np.float64)
    sum_prev = 0.0
    for i in range(n):
        forward[i] = np.exp(free_energy[i] - sum_prev)
        sum_prev += forward[i]
        if i >= footprint:
            sum_prev -= forward[i - footprint]
    backward = np.zeros(n, dtype=np.float64)
    r_forward = forward[::-1]
    sum_prod = 0.0
    for i in range(n):
        backward[i] = 1.0 - sum_prod
        sum_prod += r_forward[i] * backward[i]
        if i >= footprint:
            sum_prod -= r_forward[i - footprint] * backward[i - footprint]
    dyads = forward * backward[::-1]
    dyads = np.clip(dyads, 0, np.inf)
    dyads[np.isnan(wrapping_energy)] = 0

    occupancy = np.convolve(dyads, np.ones(footprint, dtype=np.float64), mode="same")
    occupancy = np.clip(occupancy, 0, 1)

    if show:
        plt.figure(figsize=FIGSIZE)
        plt.plot(dyads, label="Dyad", color="red")
        plt.fill_between(
            np.arange(len(occupancy)),
            occupancy,
            label="Occupancy",
            color="blue",
            alpha=0.1,
        )
        plt.ylim(-0.10, 1.1)
        plt.xlim(0, len(occupancy))
        plt.xlabel("i (bp)")
        plt.ylabel("Probability")
        plt.legend()

    return dyads, occupancy


def sample_unwrapping(
    sequence: str,
    dyad: int,
    weight: np.ndarray,
    e_contact: float = 1.0,
    steric_exclusion: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample unwrapping states around a nucleosome dyad using Boltzmann statistics.

    Models nucleosome breathing and partial unwrapping by calculating the
    thermodynamic equilibrium between 13 different unwrapping states (from -6 to +6
    contact points released). Each state has an energy cost from lost DNA-histone
    contacts and reduced wrapping energy.

    The model includes:
    - Sequential unwrapping from both ends
    - Energy penalty per lost contact point (e_contact)
    - Sequence-dependent wrapping energy changes
    - Boltzmann probability distribution over states

    Args:
        sequence: DNA sequence string.
        dyad: Nucleosome dyad position.
        weight: Dinucleotide probability weights (not log).
        e_contact: Energy per DNA-histone contact in kT units (default 1.0).
        steric_exclusion: Not currently used (legacy parameter).

    Returns:
        Tuple of (states, occupancy):
            - states: Integer array from -6 to +6 (number of contacts lost)
            - occupancy: Position-dependent occupancy probability accounting for
                        partial unwrapping

    Example:
        >>> weights = get_weight(146, period=10.0, amplitude=0.05)
        >>> states, occ = sample_unwrapping(seq, dyad=500, weights, e_contact=-0.5)
    """

    e_seg = calc_wrapping_energy(sequence, dyad, log_weights=np.log(weight)).segments

    states = np.arange(13).astype(int) - 6
    e = np.concatenate(
        (
            np.cumsum(-e_seg[:6] - e_contact)[::-1],
            [0],
            np.cumsum(-e_seg[::-1][:6] - e_contact),
        )
    )

    p = np.exp(-e)
    p[np.isnan(p)] = 0
    p /= p.sum()

    x = np.arange(-75, 75)
    occupancy = np.sum(
        [
            p[i]
            * (
                (x >= (-65 - state * 10 if state < 0 else -65))
                & (x <= (65 if state < 0 else 65 - state * 10))
            )
            for i, state in enumerate(states)
        ],
        axis=0,
    )
    if steric_exclusion > 0:
        occupancy = np.concatenate(
            (
                occupancy[steric_exclusion:75],
                np.ones(steric_exclusion * 2),
                occupancy[75:-steric_exclusion],
            )
        )

    return x, occupancy


def fetch_chromosome_sequence(filename, chromosome="II"):
    """Fetch chromosome sequence from FASTA file with automatic genome download.

    Loads a specific chromosome sequence from a multi-record FASTA file.
    If the FASTA file doesn't exist, automatically downloads the sacCer3
    (S. cerevisiae) genome using genomepy.

    Args:
        filename: Path to FASTA file (e.g., ".genomes/sacCer3/sacCer3.fa").
        chromosome: Chromosome identifier (default "II"). Accepts:
                   - Roman numerals: "I", "II", "III", etc.
                   - Arabic numerals: "1", "2", "3", etc.
                   - With prefix: "chr2", "chrII"
                   - Case-insensitive

    Returns:
        Biopython Seq object containing the chromosome sequence.

    Raises:
        FileNotFoundError: If FASTA is empty.
        ValueError: If chromosome not found in FASTA.

    Example:
        >>> seq = fetch_chromosome_sequence(".genomes/sacCer3/sacCer3.fa", "II")
        >>> print(f"Chr II length: {len(seq)} bp")

    Note:
        Automatic download only works for sacCer3 genome.
    """
    filename = Path(filename)

    # ic(str(filename.parent))
    # install genome into the parent dir if the FASTA is missing
    if not filename.exists():
        genomepy.install_genome(
            "sacCer3", "UCSC", genomes_dir=str(filename.parent.parent)
        )

    records = list(SeqIO.parse(str(filename), "fasta"))
    if not records:
        raise FileNotFoundError(f"No records in {filename}")

    # Map common Roman numerals to Arabic to broaden matching
    roman_to_arabic = {
        "i": "1",
        "ii": "2",
        "iii": "3",
        "iv": "4",
        "v": "5",
        "vi": "6",
        "vii": "7",
        "viii": "8",
        "ix": "9",
        "x": "10",
    }
    arabic_to_roman = {v: k for k, v in roman_to_arabic.items()}

    s = str(chromosome).lower().strip()
    s = s.lstrip("chr").lstrip("chromosome").strip(" _-")
    candidates = {s}
    # if s is roman, add arabic; if arabic, add roman
    if s in roman_to_arabic:
        candidates.add(roman_to_arabic[s])
    if s in arabic_to_roman:
        candidates.add(arabic_to_roman[s])
    # also try plain digits without leading zeros
    if s.isdigit():
        candidates.add(str(int(s)))

    alt = "|".join(re.escape(x) for x in sorted(candidates, key=lambda t: -len(t)))
    pat = re.compile(
        rf"(chr(?:omosome)?[ _-]?(?:{alt})|chromosome[ _-]?(?:{alt})|\b(?:{alt})\b)",
        re.I,
    )

    # choose the first record that matches our pattern, otherwise fall back to the first record
    record = next(
        (r for r in records if pat.search(r.id + " " + r.description)), records[0]
    )
    print(
        f"Selected chromosome: {str(filename.parent)}, {record.id}  len={len(record.seq)}"
    )

    # list all ORFs in the record
    for feature in record.features:
        ic(feature)
        # if feature.type == "gene":
        #     print(
        #         f"ORF: {feature.qualifiers.get('gene', ['-'])[0]} "
        #         f"{feature.location.start + 1}-{feature.location.end} "
        #         f"strand={feature.location.strand}"
        #     )
    return record.seq


class ChromatinFiber:
    """Complete framework for chromatin fiber simulation and analysis.

    This class provides a comprehensive toolkit for modeling chromatin structure,
    including DNA sequence-dependent nucleosome positioning, statistical mechanics
    calculations, Monte Carlo sampling, and DNA methylation pattern simulation.

    Key Features:
        - Sequence-dependent energy landscape calculation
        - Thermodynamic equilibrium positioning (Vanderlick formalism)
        - Stochastic sampling of nucleosome configurations
        - DNA methylation accessibility simulation
        - ORF (gene) annotation integration
        - Support for SnapGene .dna files

    Attributes:
        sequence (str): DNA sequence (uppercase).
        index (np.ndarray): Genomic position indices for each base.
        footprint (int): Nucleosome footprint size (default 146 bp).
        weight (np.ndarray): Dinucleotide probability weights for wrapping energy.
        energy (np.ndarray): Position-dependent wrapping energy landscape.
        dyad_probability (np.ndarray): Equilibrium dyad positioning probability.
        occupancy (np.ndarray): Nucleosome occupancy at each position.
        dyads (np.ndarray): Sampled nucleosome dyad positions.
        orfs (list): Annotated open reading frames/genes.
        name (str): Optional identifier for the fiber.

    Example:
        >>> # Create fiber from sequence
        >>> fiber = ChromatinFiber(sequence="ATCG"*1000, start=10000)
        >>>
        >>> # Calculate energy landscape
        >>> fiber.calc_energy_landscape(amplitude=0.05, period=10.0,
        ...                              chemical_potential=-2.0)
        >>>
        >>> # Sample configurations and simulate methylation
        >>> dyads = fiber.sample_fiber_configuration()
        >>> methylation = fiber.calc_methylation(dyads, efficiency=0.7)
        >>>
        >>> # Load from SnapGene file
        >>> fiber2 = ChromatinFiber()
        >>> fiber2.read_snapgene("plasmid.dna")

    Methods:
        calc_energy_landscape: Compute position-dependent wrapping energies.
        sample_fiber_configuration: Monte Carlo sampling of nucleosome positions.
        calc_methylation: Simulate DNA methylation with nucleosome protection.
        read_snapgene: Load sequence from SnapGene .dna file.
        fetch_orfs_by_range: Retrieve gene annotations from Ensembl.

    See Also:
        simulate_chromatin_fibers: Batch simulation with HDF5 output.
        compute_vanderlick: Statistical mechanics solver.
    """

    def __init__(self, sequence=None, start=0) -> None:
        """Initialize a chromatin fiber.

        Args:
            sequence: DNA sequence (str, Biopython Seq, or array-like).
                     If None, must be loaded later via read_snapgene().
            start: Starting genomic position (default 0).
        """
        self.footprint: int = FOOTPRINT
        self.weight: np.ndarray | None = None

        self.name: str | None = None
        if sequence is not None:
            # Accept Biopython Seq, Python str, numpy array, or similar.
            seq_str = str(sequence)
            # normalize to uppercase string for downstream processing
            self.sequence = seq_str.upper()
            self.index = np.arange(len(self.sequence), dtype=int) + start
        else:
            self.sequence: str | None = None
            self.index: np.ndarray | None = None
        self.orfs: list[dict] = []

        self.energy: np.ndarray | None = None
        self.dyad_probability: np.ndarray | None = None
        self.occupancy: np.ndarray | None = None

        self.dyads: np.ndarray = np.array([], dtype=int)

    def read_dna(self, filename: str, cut_at: int = 0) -> None:
        plasmid_data = snapgene_file_to_dict(filename)
        dyads = []
        for feature in plasmid_data["features"]:
            if "601" in feature["name"].lower():
                feature["name"] = feature["name"] = "601"
                dyads.append((feature["start"] + feature["end"]) // 2)

                self.orfs.append(
                    {
                        "chrom_acc": f"{Path(filename)}",
                        "chromosome": "plasmid",
                        "end": feature["end"],
                        "id": "-",
                        "name": feature["name"],
                        "raw_start": feature["start"],
                        "raw_stop": feature["end"],
                        "start": feature["start"],
                        "strand": 1,
                    }
                )

        dyads.sort()
        self.dyads = np.asarray(dyads, dtype=int)
        self.sequence = plasmid_data["seq"].upper()

        self.index = np.arange(len(self.sequence)).astype(int)

        self.occupancy = np.zeros(len(self.sequence))
        for dyad in self.dyads:
            start = max(0, dyad - self.footprint // 2)
            end = min(len(self.sequence), dyad + self.footprint // 2)
            self.occupancy[start:end] = 1

    def fetch_orfs_by_range(
        self,
        chromosome: str = "",
        start: int = 0,
        end: int = 0,
        organism: str = "saccharomyces cerevisiae",
    ) -> None:
        """Fetch ORF information for a genomic range.

        Parameters
        ----------
        organism : str
            Organism name (default: "saccharomyces cerevisiae")
        chromosome : str
            Chromosome name or accession number
        start : int
            Start position (1-based)
        end : int
            End position (1-based)
        """
        Entrez.email = "your.email@example.com"
        q = f"{chromosome}[Chromosome] AND {organism}[Organism] AND {start}:{end}[CHRPOS]"
        res = Entrez.read(Entrez.esearch(db="gene", term=q))
        if not res.get("IdList"):
            raise ValueError(
                f"No ORFs found for chromosome {chromosome} in range {start}:{end}"
            )

        # Loop through all gene IDs to get ORFs on both strands
        for gid in res["IdList"]:
            doc = Entrez.read(Entrez.esummary(db="gene", id=gid))["DocumentSummarySet"][
                "DocumentSummary"
            ][0]
            gi = doc.get("GenomicInfo", [{}])[0]
            raw_start = int(gi.get("ChrStart", 0))
            raw_stop = int(gi.get("ChrStop", 0))
            orf_start = min(raw_start, raw_stop) + 1
            orf_end = max(raw_start, raw_stop) + 1
            # parse/normalize strand to int
            s = gi.get("Strand")
            try:
                strand = (
                    int(s) if s is not None else (-1 if raw_start > raw_stop else 1)
                )
            except Exception:
                ss = str(s).lower() if s is not None else ""
                strand = (
                    -1
                    if ss.startswith("m") or ss.startswith("-")
                    else (
                        1
                        if ss.startswith("p") or ss.startswith("+")
                        else (-1 if raw_start > raw_stop else 1)
                    )
                )
            self.orfs.append(
                {
                    "name": doc.get("Name", "Unknown"),
                    "id": gid,
                    "chrom_acc": gi.get("ChrAccVer"),
                    "start": orf_start,
                    "end": orf_end,
                    "strand": int(strand),
                    "chromosome": doc.get("Chromosome", "Unknown"),
                    "raw_start": raw_start,
                    "raw_stop": raw_stop,
                }
            )
        return

    def fetch_orf_by_names(
        self, gene_names: str | list[str], organism: str = "saccharomyces cerevisiae"
    ) -> None:
        """Fetch ORF information for one or more genes.

        Parameters
        ----------
        gene_names : str or list[str]
            Single gene name or list of gene names to fetch
        organism : str
            Organism name (default: "saccharomyces cerevisiae")
        """
        Entrez.email = "your.email@example.com"

        # Convert single gene name to list for uniform processing
        if isinstance(gene_names, str):
            gene_names = [gene_names]

        # Process each gene
        for gene_name in gene_names:
            q = f"{gene_name}[Gene Name] AND {organism}[Organism]"
            res = Entrez.read(Entrez.esearch(db="gene", term=q))
            if not res.get("IdList"):
                raise ValueError(f"Gene {gene_name} not found for {organism}")
            gid = res["IdList"][0]
            doc = Entrez.read(Entrez.esummary(db="gene", id=gid))["DocumentSummarySet"][
                "DocumentSummary"
            ][0]
            gi = doc.get("GenomicInfo", [{}])[0]
            raw_start = int(gi.get("ChrStart", 0))
            raw_stop = int(gi.get("ChrStop", 0))
            start = min(raw_start, raw_stop) + 1
            end = max(raw_start, raw_stop) + 1
            # parse/normalize strand to int
            s = gi.get("Strand")
            try:
                strand = (
                    int(s) if s is not None else (-1 if raw_start > raw_stop else 1)
                )
            except Exception:
                ss = str(s).lower() if s is not None else ""
                strand = (
                    -1
                    if ss.startswith("m") or ss.startswith("-")
                    else (
                        1
                        if ss.startswith("p") or ss.startswith("+")
                        else (-1 if raw_start > raw_stop else 1)
                    )
                )
            self.orfs.append(
                {
                    "name": gene_name,
                    "id": gid,
                    "chrom_acc": gi.get("ChrAccVer"),
                    "start": start,
                    "end": end,
                    "strand": int(strand),
                    "chromosome": doc.get("Chromosome", "Unknown"),
                    "raw_start": raw_start,
                    "raw_stop": raw_stop,
                }
            )

    def fetch_sequence_by_range(
        self,
        chrom: str,
        start: int,
        end: int,
        organism: str = "saccharomyces cerevisiae",
    ) -> None:
        """Fetch DNA sequence for a genomic range.

        Parameters
        ----------
        chrom : str
            Chromosome name (e.g., "II") or accession number
        start : int
            Start position (1-based)
        end : int
            End position (1-based)
        organism : str
            Organism name (default: "saccharomyces cerevisiae")
        """
        Entrez.email = "your.email@example.com"

        # If chromosome name is provided (e.g., "II"), look up the accession from ORFs
        chrom_acc = chrom
        if self.orfs and not chrom.startswith("NC_"):
            # Try to find matching chromosome in ORFs
            for orf in self.orfs:
                if orf.get("chromosome") == chrom:
                    chrom_acc = orf.get("chrom_acc")
                    break

        rec = SeqIO.read(
            Entrez.efetch(
                db="nuccore",
                id=chrom_acc,
                rettype="fasta",
                seq_start=start,
                seq_stop=end,
            ),
            "fasta",
        )
        self.sequence = str(rec.seq)
        self.index = np.arange(start, end + 1).astype(int)
        self.dyads = np.array([], dtype=int)

    def fetch_sequence_by_orfs(
        self, margin_upstream: int = 0, margin_downstream: int = 0
    ) -> None:
        Entrez.email = "your.email@example.com"
        chrom = self.orfs[0].get("chrom_acc")
        end = 0
        start = 1_000_000_000

        for orf in self.orfs:
            start = min(start, int(orf["start"]))
            end = max(end, int(orf["end"]))

        start = max(1, start - margin_upstream)
        end = end + margin_downstream

        rec = SeqIO.read(
            Entrez.efetch(
                db="nuccore", id=chrom, rettype="fasta", seq_start=start, seq_stop=end
            ),
            "fasta",
        )
        self.sequence = str(rec.seq)
        self.index = np.arange(start, end + 1).astype(int)
        self.dyads = np.array([], dtype=int)

    def calc_energy_landscape(
        self,
        octamer: bool = True,
        period: float = 10.0,
        amplitude: float = 0.2,
        chemical_potential: float = -3.0,
    ) -> None:
        """Calculate position-dependent nucleosome wrapping energy landscape.

        Computes the sequence-dependent wrapping energy at every position
        along the DNA sequence using periodic dinucleotide preferences.
        Also calculates the thermodynamic equilibrium dyad probability and
        nucleosome occupancy using the Vanderlick formalism.

        Args:
            octamer: If True, use full nucleosome wrapping energy; if False,
                    use tetrasome energy (default True).
            period: DNA helical repeat periodicity in bp (default 10.0).
                   ~10 bp for B-form DNA.
            amplitude: Strength of periodic dinucleotide preference (default 0.2).
                      Range 0-0.25, typical values 0.05-0.2.
            chemical_potential: Nucleosome binding free energy in kT units (default -3.0).
                              Negative values favor nucleosome formation.

        Effects:
            Sets the following attributes:
            - self.weight: Dinucleotide probability weights
            - self.energy: Position-dependent wrapping energies
            - self.dyad_probability: Equilibrium dyad probability distribution
            - self.occupancy: Nucleosome occupancy at each position

        Example:
            >>> fiber.calc_energy_landscape(amplitude=0.05, period=10.0,
            ...                              chemical_potential=-2.0)
            >>> plt.plot(fiber.occupancy)

        Note:
            Chemical potential controls nucleosome density. More negative values
            increase nucleosome occupancy. Typical range: -5 to 0 kT.
        """
        self.weight = get_weight(
            FOOTPRINT, period=period, amplitude=amplitude, show=False
        )
        if octamer:
            self.energy = np.asarray(
                [
                    calc_wrapping_energy(
                        self.sequence, i, log_weights=np.log(self.weight)
                    ).octamer
                    for i in range(len(self.sequence))
                ]
            )
        else:
            self.energy = np.asarray(
                [
                    calc_wrapping_energy(
                        self.sequence, i, log_weights=np.log(self.weight)
                    ).tetramer
                    for i in range(len(self.sequence))
                ]
            )

        self.energy += chemical_potential

        self.dyad_probability, self.occupancy = compute_vanderlick(
            self.energy, show=False
        )

    def sample_fiber_configuration(self) -> np.ndarray:
        """Stochastically sample nucleosome positions from equilibrium distribution.

        Performs Monte Carlo sampling to generate a single-molecule realization
        of nucleosome positions. The number of nucleosomes is Poisson-distributed
        based on the equilibrium probability, and positions are sampled with
        steric exclusion to prevent overlaps.

        Returns:
            Array of sampled nucleosome dyad positions in genomic coordinates.

        Raises:
            ValueError: If dyad_probability not calculated (call calc_energy_landscape first).

        Process:
            1. Sample number of nucleosomes from Poisson distribution
            2. Iteratively sample positions from dyad probability
            3. After each sampling, exclude nearby positions (steric exclusion)
            4. Continue until all nucleosomes placed or no space remains

        Example:
            >>> fiber.calc_energy_landscape(amplitude=0.05, period=10.0)
            >>> dyads = fiber.sample_fiber_configuration()
            >>> print(f"Sampled {len(dyads)} nucleosomes")

        Note:
            Edge regions (±FOOTPRINT/2) excluded to prevent boundary artifacts.
            Each call produces a different stochastic realization.
        """
        if self.dyad_probability is None:
            raise ValueError(
                "Dyad probability must be calculated before sampling configuration"
            )
        if self.index is None:
            raise ValueError("Index must be initialized before sampling configuration")

        dyads = []

        p = self.dyad_probability.copy()
        p[np.isnan(p)] = 0

        num_nucleosomes = np.random.poisson(lam=np.sum(p))
        seq_length = len(self.dyad_probability)

        for _ in range(int(num_nucleosomes)):
            if np.sum(p) == 0:
                break
            dyads.append(np.random.choice(seq_length, p=p / p.sum()))
            nuc_start = max(0, dyads[-1] - self.footprint)
            nuc_end = min(nuc_start + self.footprint * 2, seq_length - 1)
            p[nuc_start:nuc_end] = 0

        dyads = np.sort(np.asarray(dyads))
        dyads += self.index[0]

        return dyads

    def calc_methylation(
        self,
        dyads: np.ndarray | list[int],
        e_contact: float = -1.0,
        motifs: list[str] = ["A"],
        strand: str = "both",
        efficiency: float = 0.85,
        steric_exclusion: int = 7,
    ) -> MethylationResult:
        """Simulate DNA methylation with nucleosome protection and unwrapping.

        Models DNA methylation accessibility experiments (e.g., NOMe-seq, ATAC-seq)
        by simulating:
        1. Reduced accessibility in nucleosome-wrapped regions
        2. Partial unwrapping dynamics (breathing) using Boltzmann statistics
        3. Stochastic methylation with specified efficiency
        4. Motif-specific targeting (e.g., GpC dinucleotides)

        Args:
            dyads: Array of nucleosome dyad positions (in genomic coordinates).
            e_contact: Energy per DNA-histone contact in kT units (default -1.0).
                      More negative = more stable wrapping, less breathing.
            motifs: List of DNA motifs to target for methylation (default ["A"]).
                   For NOMe-seq: ["GC"]. Case-insensitive.
            strand: Which strand to search: "both", "plus", or "minus" (default "both").
            efficiency: Probability of methylation at accessible sites (0-1, default 0.85).
            steric_exclusion: Number of contact points for exclusion calculation
                            (legacy parameter, typically 7).

        Returns:
            MethylationResult containing:
                - protected: Boolean array of nucleosome-protected positions
                - methylated: Boolean array of successfully methylated positions

        Example:
            >>> dyads = fiber.sample_fiber_configuration()
            >>> result = fiber.calc_methylation(dyads, motifs=["GC"], efficiency=0.7)
            >>> footprints = convert_to_footprints([result.methylated], fiber.index)

        Note:
            Protection probability accounts for nucleosome unwrapping dynamics.
            Positions can be protected but not methylated if efficiency < 1.
        """

        if self.sequence is None:
            raise ValueError(
                "Sequence must be initialized before calculating methylation"
            )
        if self.weight is None:
            raise ValueError(
                "Weight must be initialized before calculating methylation"
            )
        if self.index is None:
            raise ValueError("Index must be initialized before calculating methylation")

        methylation_targets = np.zeros(len(self.sequence), dtype=int)
        for motif in motifs:
            offset = 1 if motif == "GC" else 0
            if strand in ["both", "forward", "+"]:
                for match in re.finditer(motif, self.sequence):
                    methylation_targets[match.start() + offset] = 1
            if strand in ["both", "reverse", "-"]:
                rev_motif = motif[::-1].translate(str.maketrans("ACGT", "TGCA"))
                for match in re.finditer(rev_motif, self.sequence):
                    methylation_targets[match.start() + offset] = 1

        protected = np.zeros(len(self.sequence)).astype(np.float64)
        for dyad in dyads:
            x, occupancy = sample_unwrapping(
                self.sequence,
                dyad - self.index[0],
                self.weight,
                e_contact=e_contact,
                steric_exclusion=steric_exclusion,
            )
            protected[x + dyad - self.index[0] - 1] += occupancy

        protected = np.clip(protected, 0, 1)
        p_methylated = (1 - protected) * methylation_targets * efficiency
        methylated = np.random.binomial(1, p_methylated).astype(float)

        # remove non-target bases from methylation array
        methylated[methylation_targets == 0] = np.nan

        # return float array (with NaN for non-targets) to avoid invalid cast warnings
        return MethylationResult(protected, methylated)

    def encode_methylations(self, methylated: np.ndarray) -> str:
        """make methylated bases lowercase in sequence string"""
        if self.sequence is None:
            raise ValueError(
                "Sequence must be initialized before encoding methylations"
            )

        encoded_sequence = [
            base.lower() if methylated[i] == 1 else base.upper()
            for i, base in enumerate(self.sequence)
        ]
        return "".join(encoded_sequence)


def simulate_chromatin_fibers(params: SimulationParams, filename: str):
    """Run batch chromatin fiber simulation with HDF5 output.

    Generates multiple random DNA sequences, simulates nucleosome positioning
    and methylation patterns for each, and incrementally saves results to
    an HDF5 file for efficient storage and later analysis.

    The simulation workflow for each fiber:
        1. Generate random DNA sequence
        2. Calculate energy landscape
        3. Sample nucleosome configuration
        4. Simulate methylation with protection
        5. Write to HDF5 file

    Args:
        params: SimulationParams object containing all simulation parameters:
            - n_samples: Number of fibers to simulate
            - length_bp: DNA sequence length
            - amplitude: Dinucleotide preference strength
            - period_bp: DNA bending periodicity
            - chemical_potential_kT: Nucleosome binding energy
            - e_contact_kT: Contact energy for unwrapping
            - motifs: Methylation target motifs
            - strand: Strand specification
            - efficiency: Methylation efficiency
            - steric_exclusion_bp: Footprint exclusion size
        filename: Output HDF5 filename (without extension).

    Returns:
        str: Path to created HDF5 file.

    HDF5 Structure:
        Datasets:
            - dyad_flat: Concatenated dyad positions (int64)
            - dyad_lengths: Length of dyad array per sample (int64)
            - encoded_flat: Concatenated encoded sequences (int8)
            - encoded_lengths: Length of sequence per sample (int64)
            - methylated_sequences: Variable-length strings with methylation
        Attributes:
            - n_samples: Total number of simulated fibers
            - Plus all SimulationParams as attributes

    Example:
        >>> params = SimulationParams(
        ...     n_samples=1000,
        ...     length_bp=10_000,
        ...     amplitude=0.05,
        ...     chemical_potential_kT=-2.0
        ... )
        >>> file = simulate_chromatin_fibers(params, "results/batch1")
        >>> data = read_simulation_results(file)

    Note:
        Results are written incrementally to handle large datasets.
        Progress shown via tqdm progress bar.

    See Also:
        read_simulation_results: Load and parse saved simulations.
        SimulationParams: Parameter configuration dataclass.
    """

    h5_context = _H5Writer(filename, params)

    print("Save fibers in HDF5 file:", h5_context.filename)
    for i in tqdm(range(params.n_samples), desc="Simulating fibers"):
        sequence = "".join(np.random.choice(list("ACGT"), size=params.length_bp))
        fiber = ChromatinFiber(sequence=sequence)
        fiber.calc_energy_landscape(
            octamer=True,
            amplitude=params.amplitude,
            chemical_potential=params.chemical_potential_kT,
            period=params.period_bp,
        )

        dyads = fiber.sample_fiber_configuration()

        methylation = fiber.calc_methylation(
            dyads,
            e_contact=params.e_contact_kT,
            motifs=params.motifs,
            strand=params.strand,
            efficiency=params.efficiency,
            steric_exclusion=params.steric_exclusion_bp,
        )

        methylated_seq = fiber.encode_methylations(methylation.methylated)
        encoded_seq = encode_seq(methylated_seq)

        h5_context.write_sample(dyads, encoded_seq, methylated_seq)

    file_out = h5_context.close().__str__()
    return file_out


class _H5Writer:
    """Context manager for incremental writing to HDF5 file."""

    def __init__(self, filename: str, params: SimulationParams):
        self.filename = Path(filename).with_suffix(".h5")

        # Create parent directory if it doesn't exist
        self.filename.parent.mkdir(parents=True, exist_ok=True)

        self.file_exists = self.filename.exists()
        mode = "a" if self.file_exists else "w"
        self.f = h5py.File(self.filename, mode)

        # Variable-length string dtype
        self.dt_vlen_str = h5py.special_dtype(vlen=str)

        # Initialize datasets if new file
        if not self.file_exists or "dyad_flat" not in self.f:
            # Save simulation parameters as attributes
            self.f.attrs["n_samples"] = params.n_samples
            self.f.attrs["length_bp"] = params.length_bp
            self.f.attrs["amplitude"] = params.amplitude
            self.f.attrs["period_bp"] = params.period_bp
            self.f.attrs["chemical_potential_kT"] = params.chemical_potential_kT
            self.f.attrs["e_contact_kT"] = params.e_contact_kT
            # Convert list/tuple to string for motifs
            self.f.attrs["motifs"] = (
                ",".join(params.motifs)
                if isinstance(params.motifs, (list, tuple))
                else str(params.motifs)
            )
            self.f.attrs["strand"] = params.strand
            self.f.attrs["efficiency"] = params.efficiency
            self.f.attrs["steric_exclusion_bp"] = params.steric_exclusion_bp
            self.f.create_dataset(
                "dyad_flat",
                shape=(0,),
                maxshape=(None,),
                dtype="i8",
                chunks=True,
                compression="gzip",
                compression_opts=4,
            )
            self.f.create_dataset(
                "dyad_lengths", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
            )
            self.f.create_dataset(
                "encoded_flat",
                shape=(0,),
                maxshape=(None,),
                dtype="i1",
                chunks=True,
                compression="gzip",
                compression_opts=4,
            )
            self.f.create_dataset(
                "encoded_lengths", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
            )
            self.f.create_dataset(
                "methylated_sequences",
                shape=(0,),
                maxshape=(None,),
                dtype=self.dt_vlen_str,
                chunks=True,
            )
            self.f.attrs["n_samples"] = 0

    def write_sample(
        self, dyads: np.ndarray, encoded: np.ndarray, methylated: str
    ) -> None:
        """Write a single sample to the HDF5 file."""
        # Append dyad data
        dyad_ds = self.f["dyad_flat"]
        old_dyad_size = dyad_ds.shape[0]
        dyad_ds.resize((old_dyad_size + len(dyads),))
        dyad_ds[old_dyad_size:] = dyads

        dyad_len_ds = self.f["dyad_lengths"]
        old_len_size = dyad_len_ds.shape[0]
        dyad_len_ds.resize((old_len_size + 1,))
        dyad_len_ds[old_len_size] = len(dyads)

        # Append encoded data
        enc_ds = self.f["encoded_flat"]
        old_enc_size = enc_ds.shape[0]
        enc_ds.resize((old_enc_size + len(encoded),))
        enc_ds[old_enc_size:] = encoded

        enc_len_ds = self.f["encoded_lengths"]
        old_enc_len_size = enc_len_ds.shape[0]
        enc_len_ds.resize((old_enc_len_size + 1,))
        enc_len_ds[old_enc_len_size] = len(encoded)

        # Append methylated sequence
        meth_ds = self.f["methylated_sequences"]
        old_meth_size = meth_ds.shape[0]
        meth_ds.resize((old_meth_size + 1,))
        meth_ds[old_meth_size] = methylated

        # Update sample count
        self.f.attrs["n_samples"] = int(dyad_len_ds.shape[0])

    def close(self) -> None:
        """Close the HDF5 file and print summary."""
        n_samples = int(self.f.attrs["n_samples"])
        self.f.close()
        return self.filename


def read_simulation_results(filename: str, indices=None):
    """Load simulation results or parameters from HDF5 file.

    Flexible reader that returns different data based on the indices parameter.
    Can retrieve simulation parameters, individual samples, or batches of samples.

    Args:
        filename: Path to HDF5 file (with or without .h5 extension).
        indices: What to retrieve:
            - None: Return SimulationParams (metadata only)
            - int: Return single sample at that index
            - list/array: Return batch of samples at those indices

    Returns:
        Depends on indices parameter:
        - None → SimulationParams object
        - int → Tuple (dyad_positions, encoded_sequence, methylated_sequence)
        - list → Tuple of lists ([dyads], [encoded_seqs], [methylated_seqs])

    Raises:
        IndexError: If requested index out of range.
        FileNotFoundError: If HDF5 file doesn't exist.

    Examples:
        >>> # Get simulation parameters
        >>> params = read_simulation_results("data/batch1.h5")
        >>> print(f"Simulated {params.n_samples} fibers")

        >>> # Get single sample
        >>> dyads, encoded, methylated = read_simulation_results("data/batch1.h5", 0)
        >>> print(f"Sample has {len(dyads)} nucleosomes")

        >>> # Get batch of samples
        >>> batch = read_simulation_results("data/batch1.h5", [0, 5, 10])
        >>> dyads_list, encoded_list, meth_list = batch
        >>> print(f"Loaded {len(dyads_list)} samples")

    Note:
        Encoded sequences use numeric encoding: A=0, C=1, G=2, T=3.
        Methylated sequences are strings with uppercase=unmethylated, lowercase=methylated.

    See Also:
        simulate_chromatin_fibers: Generate simulations.
        ChromatinFiber.decode_sequence: Convert encoded sequences back to strings.
    """
    filename = Path(filename).with_suffix(".h5")

    with h5py.File(filename, "r") as f:
        # Case 1: Return parameters
        if indices is None:
            motifs_str = f.attrs.get("motifs", "A")
            motifs = (
                tuple(motifs_str.split(",")) if "," in motifs_str else (motifs_str,)
            )

            return SimulationParams(
                n_samples=int(f.attrs.get("n_samples", 0)),
                length_bp=int(f.attrs.get("length_bp", 10000)),
                amplitude=float(f.attrs.get("amplitude", 0.05)),
                period_bp=float(f.attrs.get("period_bp", 10.0)),
                chemical_potential_kT=float(f.attrs.get("chemical_potential_kT", 0.0)),
                e_contact_kT=float(f.attrs.get("e_contact_kT", -0.5)),
                motifs=list(motifs),
                strand=str(f.attrs.get("strand", "both")),
                efficiency=float(f.attrs.get("efficiency", 0.7)),
                steric_exclusion_bp=int(f.attrs.get("steric_exclusion_bp", 0)),
            )

        n_samples = int(f.attrs["n_samples"])

        # Case 2: Single sample (integer index)
        if isinstance(indices, int):
            if indices < 0 or indices >= n_samples:
                raise IndexError(f"Index {indices} out of range [0, {n_samples})")

            # Read lengths to compute offsets
            dyad_lengths = f["dyad_lengths"][:]
            enc_lengths = f["encoded_lengths"][:]

            # Compute offsets
            dyad_offsets = np.concatenate([[0], np.cumsum(dyad_lengths[:-1])])
            enc_offsets = np.concatenate([[0], np.cumsum(enc_lengths[:-1])])

            # Read slices
            dyad_start = int(dyad_offsets[indices])
            dyad_end = int(dyad_start + dyad_lengths[indices])
            dyads = f["dyad_flat"][dyad_start:dyad_end]

            enc_start = int(enc_offsets[indices])
            enc_end = int(enc_start + enc_lengths[indices])
            encoded = f["encoded_flat"][enc_start:enc_end]

            methylated = f["methylated_sequences"][indices]

            return dyads, encoded, methylated

        # Case 3: Batch (list or array of indices)
        indices_arr = np.asarray(indices)
        if np.any(indices_arr < 0) or np.any(indices_arr >= n_samples):
            raise IndexError(f"Some indices out of range [0, {n_samples})")

        # Read all lengths once
        dyad_lengths = f["dyad_lengths"][:]
        enc_lengths = f["encoded_lengths"][:]

        # Compute offsets
        dyad_offsets = np.concatenate([[0], np.cumsum(dyad_lengths)])
        enc_offsets = np.concatenate([[0], np.cumsum(enc_lengths)])

        dyads_list = []
        encoded_list = []
        methylated_list = []

        for idx in indices_arr:
            dyad_start = int(dyad_offsets[idx])
            dyad_end = int(dyad_offsets[idx + 1])
            dyads_list.append(f["dyad_flat"][dyad_start:dyad_end])

            enc_start = int(enc_offsets[idx])
            enc_end = int(enc_offsets[idx + 1])
            encoded_list.append(f["encoded_flat"][enc_start:enc_end])

            methylated_list.append(f["methylated_sequences"][idx])

        return dyads_list, encoded_list, methylated_list


if __name__ == "__main__":
    if False:
        fiber = ChromatinFiber()
        plotter = Plotter()

        fiber.read_dna(r"data/S_CP115_pUC18 (Amp) 16x167.dna", 1300)
        # plotter.plot(fiber)
        # plt.show()

        fiber.orfs.clear()
        fiber.fetch_orf_by_names(
            ["GAL7", "GAL10", "FUR4", "GAL1"], organism="saccharomyces cerevisiae"
        )

        handle = 1000

        fiber.fetch_orfs_by_range(
            fiber.orfs[0]["chromosome"],
            fiber.orfs[0]["start"] - handle,
            fiber.orfs[0]["end"] + handle,
        )

        fiber.fetch_sequence_by_orfs(margin_upstream=1000, margin_downstream=1000)

        # plotter.plot(cf, occupancy=True, dyads=False, orfs=True)

        # get_weight(FOOTPRINT, period=10.1, amplitude=0.2, show=True)

        # fiber.fetch_orfs_by_range("II", 273_253, 284_607)
        # fiber.fetch_sequence_by_range("II", 273_253, 284_607)
        # fiber.fetch_sequence_by_range("II", 277_900, 280_000)
        # fiber.fetch_sequence_by_orfs(margin_upstream=2000, margin_downstream=2000)

        fiber.calc_energy_landscape(
            octamer=True, period=10.0, amplitude=0.05, chemical_potential=1.5
        )

        dyads_sampled, occupancy_sampled = fiber.sample_fiber_configuration()

        plotter.plot(fiber, occupancy=True, dyads=dyads_sampled, orfs=True, energy=True)

        methylation = fiber.calc_methylation(
            dyads=dyads_sampled,
            e_contact=-0.6,
            motifs=["A"],
            strand="both",
            efficiency=0.85,
        )

        plt.plot(fiber.index, methylation.protected, label="Protected", color="blue")

        plt.plot(
            fiber.index,
            methylation.methylated,
            label="Methylated",
            color="green",
            linestyle="none",
            marker="o",
            markersize=2,
            alpha=0.4,
        )

        n = 100
        mean_protected = np.zeros(len(fiber.sequence))
        for _ in range(n):
            dyads_sampled, occupancy_sampled = fiber.sample_fiber_configuration()
            methylation = fiber.calc_methylation(
                dyads=dyads_sampled,
                e_contact=-0.6,
                motifs=["A"],
                strand="both",
                efficiency=0.85,
            )
            mean_protected += methylation.protected
        mean_protected /= n

        plt.plot(fiber.index, mean_protected, label="Mean Protected", color="green")

        plt.show()

    fiber = ChromatinFiber()
    fiber.read_dna(r"data/S_CP115_pUC18 (Amp) 16x167.dna", 1300)

    fiber.calc_energy_landscape(octamer=False, amplitude=0.1, chemical_potential=1)

    dyads = fiber.sample_fiber_configuration()

    methylation = fiber.calc_methylation(dyads, steric_exclusion=10)

    # Test incremental HDF5 writing
    params = SimulationParams(n_samples=3, length_bp=500)
    x, y, z = simulate_chromatin_fibers(params, filename="test_incremental.h5")
    ic(x[0])
    ic(y[0])
    ic(z[0])

    # Test appending more samples
    params2 = SimulationParams(n_samples=2, length_bp=500)
    simulate_chromatin_fibers(params2, filename="test_incremental.h5")
