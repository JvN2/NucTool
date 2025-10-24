from dataclasses import dataclass
import numpy as np
import pandas as pd
import re
from pathlib import Path
from snapgene_reader import snapgene_file_to_dict
import matplotlib.pyplot as plt
from icecream import ic
from Bio import Entrez, SeqIO
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import genomepy


FOOTPRINT = 146
FIGSIZE = (13, 4)

_ASCII_TO_IDX = np.full(256, -1, dtype=np.int8)
_ASCII_TO_IDX[ord("A")] = 0
_ASCII_TO_IDX[ord("C")] = 1
_ASCII_TO_IDX[ord("G")] = 2
_ASCII_TO_IDX[ord("T")] = 3


@dataclass(frozen=True)
class WrappingEnergyResult:
    octamer: float
    tetramer: float
    segments: np.ndarray


@dataclass(frozen=True)
class MethylationResult:
    protected: np.ndarray
    methylated: np.ndarray


def plot_footprints(footprints, index):
    def create_cmap(crange=(0, 250)):
        colors = [
            (0, "white"),
            (10, "whitesmoke"),
            (30, "magenta"),
            (45, "blue"),
            (55, "blue"),
            (90, "cyan"),
            (100, "cyan"),
            (132, "lime"),
            (180, "limegreen"),
            (250, "darkgreen"),
        ]
        colors = [(x / colors[-1][0], c) for x, c in colors]

        # Define the new colors
        norm = mcolors.Normalize(vmin=crange[0], vmax=crange[1])
        cmap = mcolors.LinearSegmentedColormap.from_list(
            name="Sterachis", colors=colors, N=250  # Number of color steps
        )
        return cmap

    def plot_box(ax, xmin, xmax, ymin, ymax, color, alpha=1):
        rectangle = mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, facecolor=color, alpha=alpha
        )
        ax.add_patch(rectangle)
        ax.patch.set_zorder(2)

    ax = plt.gca()
    cmap = create_cmap()

    ids = footprints["read_id"].unique()
    xlim = (index[0], index[-1])

    plt.hlines(ids, color="lightgrey", *xlim, zorder=1)
    norm = Normalize(0, 250)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    crange = (-10, 250)

    for i, id in tqdm(enumerate(ids), desc="Plotting footprints"):
        for _, row in footprints[footprints["read_id"] == id].iterrows():
            plot_box(
                ax,
                row["start"],
                row["end"],
                i - 0.3,
                i + 0.3,
                cmap(norm(np.clip(row["width"], *crange))),
            )

    plt.xlim(xlim)
    plt.ylim(-0.5, len(ids) + 0.5)

    plt.yticks([])
    plt.box(False)
    plt.gca().spines["left"].set_visible(False)
    plt.xlabel("i (bp)")

    norm = mcolors.Normalize(vmin=0, vmax=250)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, ticks=np.linspace(0, 250, 6))
    plt.gcf().set_size_inches(14.5, 3)

    plt.show()
    return


def convert_to_footprints(methylated, index, minimal_footprint=10):
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
    """Convert sequence string to numeric indices for vectorized computation."""
    # Convert Biopython Seq object to string if needed
    seq_str = str(seq)
    b = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
    return _ASCII_TO_IDX[b]


def get_weight(
    w: int, period: float, amplitude: float, show: bool = False
) -> np.ndarray:
    """Generate periodic dinucleotide weights to model DNA bendability."""
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
    start = -len(log_weights[0, 0]) // 2
    end = start + len(log_weights[0, 0]) + 1
    if dyad + start < 0 or dyad + end - 1 >= len(sequence):
        return WrappingEnergyResult(
            octamer=np.nan, tetramer=np.nan, segments=np.full(14, np.nan)
        )

    # Extract and encode sequence segment
    seq_segment = sequence[dyad + start : dyad + end - 1]
    idx = encode_seq(seq_segment)

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
    wrapping_energy: np.ndarray, mu: float, show: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Compute equilibrium dyad probability accounting for steric exclusion between nucleosomes."""

    free_energy = wrapping_energy + mu
    free_energy = np.nan_to_num(free_energy, nan=np.nanmax(free_energy))
    free_energy *= -1

    n = free_energy.size
    forward = np.zeros(n, dtype=np.float64)
    sum_prev = 0.0
    for i in range(n):
        forward[i] = np.exp(free_energy[i] - sum_prev)
        sum_prev += forward[i]
        if i >= FOOTPRINT:
            sum_prev -= forward[i - FOOTPRINT]
    backward = np.zeros(n, dtype=np.float64)
    r_forward = forward[::-1]
    sum_prod = 0.0
    for i in range(n):
        backward[i] = 1.0 - sum_prod
        sum_prod += r_forward[i] * backward[i]
        if i >= FOOTPRINT:
            sum_prod -= r_forward[i - FOOTPRINT] * backward[i - FOOTPRINT]
    dyads = forward * backward[::-1]
    dyads = np.clip(dyads, 0, np.inf)
    dyads[np.isnan(wrapping_energy)] = 0

    occupancy = np.convolve(dyads, np.ones(FOOTPRINT, dtype=np.float64), mode="same")
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
    """Sample unwrapping states around a nucleosome dyad."""

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
    """Return the sequence for `chromosome` from a multi-record FASTA.
    If the FASTA is missing and genomepy is available, attempt to download the sacCer3 genome into the
    parent directory of `filename` (so filename should point to the expected .fa path).
    """
    filename = Path(filename)

    ic(str(filename.parent))
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
    print(f"Selected record: {record.id}  len={len(record.seq)}")

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
    """Simple container and utilities for a linear chromatin fiber.

    Responsibilities:
    - store DNA sequence and dyad positions
    - compute occupancy array for a given nucleosome footprint
    - sample a single-molecule configuration by stochastic sampling of dyads
    """

    def __init__(self, sequence=None, start=0) -> None:
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

        self.dyad_probability, self.occupancy = compute_vanderlick(
            self.energy, mu=chemical_potential, show=False
        )

    def sample_fiber_configuration(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Stochastically sample a single nucleosome arrangement to generate ensemble statistics.
        Edge regions are excluded to prevent boundary artifacts from biasing occupancy.
        """
        if self.dyad_probability is None:
            raise ValueError(
                "Dyad probability must be calculated before sampling configuration"
            )
        if self.index is None:
            raise ValueError("Index must be initialized before sampling configuration")

        dyads = []
        occupancy = np.zeros_like(self.dyad_probability)

        p = self.dyad_probability.copy()
        p[np.isnan(p)] = 0
        num_nucleosomes = np.random.poisson(lam=np.sum(p))

        half_footprint = self.footprint // 2
        seq_length = len(self.dyad_probability)

        for _ in range(int(num_nucleosomes)):
            dyads.append(np.random.choice(seq_length, p=p / p.sum()))

            nuc_start = dyads[-1] - half_footprint
            nuc_end = nuc_start + self.footprint

            p[nuc_start:nuc_end] = 0
            occupancy[nuc_start:nuc_end] = 1

        dyads = np.sort(np.asarray(dyads))
        dyads += self.index[0]

        return dyads, occupancy

    def calc_methylation(
        self,
        dyads: np.ndarray | list[int],
        e_contact: float = -1.0,
        motifs: list[str] = ["A"],
        strand: str = "both",
        efficiency: float = 0.85,
        steric_exclusion: int = 7,
    ) -> MethylationResult:

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
                self.sequence, dyad - self.index[0], self.weight, e_contact=e_contact
            )
            protected[x + dyad - self.index[0] - 1] += occupancy

        protected = np.clip(protected, 0, 1)
        p_methylated = (1 - protected) * methylation_targets * efficiency
        methylated = np.random.binomial(1, p_methylated).astype(float)

        # remove non-target bases from methylation array
        methylated[methylation_targets == 0] = np.nan

        # return float array (with NaN for non-targets) to avoid invalid cast warnings
        return MethylationResult(protected, methylated)


class SequencePlotter:
    def __init__(self) -> None:
        self.figure_counter: int = 1
        self.fig_size: tuple[int, int] = FIGSIZE
        self.font_size: int = 14

        plt.rcParams["font.family"] = "serif"
        plt.rcParams.update({"axes.titlesize": self.font_size})
        plt.rcParams.update({"axes.labelsize": self.font_size})
        plt.rcParams.update({"xtick.labelsize": self.font_size * 0.83})
        plt.rcParams.update({"ytick.labelsize": self.font_size * 0.83})
        plt.rcParams.update({"legend.fontsize": self.font_size * 0.83})

    def add_caption(self, title: str, fig_num: int | None = None) -> None:
        if fig_num is None:
            fig_num = self.figure_counter
        plt.tight_layout()
        formatted_caption = f"$\\bf{{Figure\\ {fig_num})}}$ {title}"
        plt.suptitle(
            formatted_caption,
            x=0,
            y=-0.025,
            ha="left",
            fontsize=self.font_size,
            wrap=True,
        )
        self.figure_counter += 1
        return

    def plot(
        self,
        fiber: ChromatinFiber,
        occupancy: bool = True,
        dyads: bool | np.ndarray = True,
        orfs: bool = False,
        energy: bool = False,
        methylation: bool = False,
    ) -> None:

        plt.figure(figsize=(12, 2))
        plt.xlabel("i (bp)")
        plt.ylabel("occupancy")
        plt.xlim(min(fiber.index), max(fiber.index))
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.subplots_adjust(right=0.94)

        if isinstance(occupancy, bool) and fiber.occupancy is not None:
            plt.fill_between(fiber.index, fiber.occupancy, color="blue", alpha=0.3)
        elif isinstance(occupancy, np.ndarray):
            plt.fill_between(fiber.index, occupancy, color="blue", alpha=0.3)

        if isinstance(dyads, bool):
            if dyads:
                for d in fiber.dyads:
                    plt.axvline(x=d, ymin=0, color="grey", linestyle="--", alpha=0.7)
        else:
            for d in dyads:
                plt.axvline(x=d, ymin=0, color="grey", linestyle="--", alpha=0.7)

        if orfs:
            for orf in fiber.orfs:
                name = orf["name"]
                if orf["strand"] == -1:
                    name = f"< {name}"
                    top = 0
                    bottom = -0.1
                else:
                    name = f"{name} >"
                    top = 0
                    bottom = -0.1

                start = min(orf["start"], orf["end"])
                end = max(orf["start"], orf["end"])
                plt.fill_between(
                    [start, end],
                    bottom,
                    top,
                    color="blue",
                    alpha=0.5,
                    label=orf["name"],
                )

                plt.text(
                    (start + end) / 2,
                    -0.06,
                    name,
                    ha="center",
                    va="center",
                    fontsize=7,
                    font="arial",
                    weight="bold",
                    color="white",
                )

        if energy and fiber.energy is not None:
            ax1 = plt.gca()  # Get current axis (left y-axis)
            ax2 = plt.twinx()
            ax2.plot(fiber.index, fiber.energy, color="red", linewidth=0.5)
            ax2.set_ylabel(
                "energy (k$_B$T)", rotation=270, labelpad=18, loc="center", color="red"
            )
            ax2.set_ylim(np.nanmin(fiber.energy) * 1.3, np.nanmax(fiber.energy) * 2.6)
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.grid(False)
            plt.sca(ax1)  # Make left y-axis active again

        if isinstance(methylation, np.ndarray):
            plt.plot(
                fiber.index, methylation, "o", color="green", markersize=2, alpha=0.5
            )

    def save_figure(self, filename: str) -> None:
        plt.savefig(
            f"figures/figure_{self.figure_counter}.png", dpi=300, bbox_inches="tight"
        )


if __name__ == "__main__":
    if False:
        fiber = ChromatinFiber()
        plotter = SequencePlotter()

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

    fiber.calc_energy_landscape(octamer=False, amplitude=0.1, chemical_potential=10)

    molecule = fiber.sample_fiber_configuration()
    methylation = fiber.calc_methylation(molecule[0])

    plotter = SequencePlotter()
    plotter.plot(
        fiber,
        occupancy=methylation.protected,
        energy=True,
        methylation=methylation.methylated,
    )
    plt.show()
