from dataclasses import dataclass
import numpy as np
from pathlib import Path
from snapgene_reader import snapgene_file_to_dict
import matplotlib.pyplot as plt
from icecream import ic
from Bio import Entrez, SeqIO


FOOTPRINT = 146
FIGSIZE = (12, 3)

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


def encode_seq(seq: str) -> np.ndarray:
    """Convert sequence string to numeric indices for vectorized computation."""
    # Convert Biopython Seq object to string if needed
    seq_str = str(seq)
    b = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
    return _ASCII_TO_IDX[b]


def get_weight(w: int, period: float, amplitude: float, show=False) -> np.ndarray:
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
            octamer=np.nan, tetramer=np.nan, segments=np.array(np.nan)
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


def compute_vanderlick(wrapping_energy: np.ndarray, mu: float, show: bool = False):
    """Compute equilibrium dyad probability accounting for steric exclusion between nucleosomes."""

    free_energy = wrapping_energy - mu
    # fill nans with zeros
    free_energy = np.nan_to_num(free_energy, nan=np.nanmax(free_energy))

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


class ChromatinFiber:
    """Simple container and utilities for a linear chromatin fiber.

    Responsibilities:
    - store DNA sequence and dyad positions
    - compute occupancy array for a given nucleosome footprint
    - sample a single-molecule configuration by stochastic sampling of dyads
    """

    def __init__(self):
        """Initialize the ChromatinFiber.

        Parameters
        - sequence: optional DNA sequence (string). If provided, `self.index`
          will be created as np.arange(len(sequence)).
        - footprint: nucleosome footprint in bp (default 146).
        """

        self.footprint = FOOTPRINT
        self.name = None
        self.sequence = None
        self.index = None
        self.dyads = np.array([], dtype=int)
        self.energy = None
        self.occupancy = None
        self.orfs = []

    def read_dna(self, filename: str, cut_at: int = 0):
        plasmid_data = snapgene_file_to_dict(filename)
        dyads = []
        for feature in plasmid_data["features"]:
            if "601" in feature["name"].lower():
                dyads.append((feature["start"] + feature["end"]) // 2)

        dyads.sort()
        self.dyads = np.asarray(dyads, dtype=int)
        self.sequence = plasmid_data["seq"].upper()

        self.index = np.arange(len(self.sequence)).astype(int)

        self.occupancy = np.zeros(len(self.sequence))
        for dyad in self.dyads:
            start = max(0, dyad - self.footprint // 2)
            end = min(len(self.sequence), dyad + self.footprint // 2)
            self.occupancy[start:end] = 1

        # if cut_at > 0:
        #     self.sequence = self.sequence[cut_at:] + self.sequence[:cut_at]
        #     dyads = (np.asarray(dyads) - cut_at).tolist()
        #     dyads = [dyad + len(self.sequence) if dyad < 0 else dyad for dyad in dyads]
        #     self.index = self.index[cut_at:].tolist() + self.index[:cut_at].tolist()

        return

    def fetch_orf(
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

    def fetch_orf_sequence(
        self, margin_upstream: int = 0, margin_downstream: int = 0
    ) -> tuple[np.ndarray, str]:
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
        self.sequence = rec.seq
        self.index = np.arange(start, end + 1).astype(int)
        self.dyads = np.array([], dtype=int)

    def calc_energy_landscape(
        self, octamer=True, period=10.0, amplitude=0.2, chemical_potential=-3.0
    ):
        weight = get_weight(FOOTPRINT, period=period, amplitude=amplitude, show=False)
        if octamer:
            self.energy = np.asarray(
                [
                    calc_wrapping_energy(
                        self.sequence, i, log_weights=np.log(weight)
                    ).octamer
                    for i in range(len(self.sequence))
                ]
            )
        else:
            self.energy = np.asarray(
                [
                    calc_wrapping_energy(
                        self.sequence, i, log_weights=np.log(weight)
                    ).tetramer
                    for i in range(len(self.sequence))
                ]
            )

        self.dyad_probability, self.occupancy = compute_vanderlick(
            self.energy, mu=chemical_potential, show=False
        )


class SequencePlotter:
    def __init__(self):
        self.figure_counter = 1
        self.fig_size = FIGSIZE
        self.font_size = 14

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
        dyads: bool = True,
        orfs: bool = False,
        energy: bool = False,
    ) -> None:

        plt.figure(figsize=(12, 2))
        plt.xlabel("i (bp)")
        plt.ylabel("occupancy")
        plt.xlim(min(fiber.index), max(fiber.index))
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()

        if occupancy and fiber.occupancy is not None:
            plt.fill_between(fiber.index, fiber.occupancy, color="blue", alpha=0.3)

        if dyads:
            for d in fiber.dyads:
                plt.axvline(x=d, color="grey", linestyle="--", alpha=0.7)

        if orfs:
            for orf in fiber.orfs:
                start = min(orf["start"], orf["end"])
                end = max(orf["start"], orf["end"])
                plt.fill_between(
                    [start, end],
                    -0.1,
                    0,
                    color="blue",
                    alpha=0.5,
                    label=orf["name"],
                )
                name = orf["name"]
                if orf["strand"] == -1:
                    name = f"< {name}"
                elif orf["strand"] == 1:
                    name = f"{name} >"

                plt.text(
                    (start + end) / 2,
                    -0.05,
                    name,
                    ha="center",
                    va="center",
                    fontsize=7,
                    font="arial",
                    weight="bold",
                    color="white",
                )

        if energy and fiber.energy is not None:
            plt.twinx()
            plt.plot(fiber.index, fiber.energy, color="red", linewidth=0.5)
            plt.ylabel("energy", color="red")
            plt.ylim(np.nanmin(fiber.energy), np.nanmax(fiber.energy))
            plt.tick_params(axis="y", labelcolor="red")
            plt.yticks(fontsize=7)
            plt.grid(False)

        plt.show()

    def save_figure(self, filename: str) -> None:
        plt.savefig(
            f"figures/figure_{self.figure_counter}.png", dpi=300, bbox_inches="tight"
        )


if __name__ == "__main__":
    cf = ChromatinFiber()
    plotter = SequencePlotter()

    # cf.read_dna(r"data/S_CP115_pUC18 (Amp) 16x167.dna", 1300)
    # plotter.plot(cf)

    cf.fetch_orf(["GAL10", "GAL1"])
    cf.fetch_orf_sequence(margin_upstream=1000, margin_downstream=1000)

    # plotter.plot(cf, occupancy=True, dyads=False, orfs=True)

    # get_weight(FOOTPRINT, period=10.1, amplitude=0.2, show=True)
    cf.calc_energy_landscape(
        octamer=True, period=10.0, amplitude=0.2, chemical_potential=-12.0
    )
    plotter.plot(cf, occupancy=True, dyads=False, orfs=True, energy=True)

    plt.show()
