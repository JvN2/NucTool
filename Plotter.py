"""
Plotter module for creating publication-quality figures with automatic formatting.

Contains two main classes:
- Plotter: Modern plotting utility with automatic panel labeling and caption management
- SequencePlotter: Legacy class for chromatin fiber visualization (same interface as Plotter)

Key features:
- Automatic subplot creation and labeling (A, B, C, D...)
- Extract subplot titles and move them to formatted captions
- Consistent layout with constrained_layout by default
- Publication-ready figure captions with "Figure X)" formatting
"""

import numpy as np
import matplotlib

# matplotlib.use("Agg")

import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ChromatinFibers import ChromatinFiber

FIGSIZE = (13, 4)


class Plotter:
    """
    Modern plotting utility for creating publication-quality figures.

    Features:
    - Automatic subplot creation with consistent formatting
    - Automatic panel labeling (A, B, C, D...)
    - Extract and format panel titles into captions
    - Constrained layout for consistent plot areas
    - Publication-ready figure captions

    Typical workflow:
        plot = Plotter()
        plot.new(nrows=2, ncols=2)
        for panel in plot.panels:
            panel.plot(x, y)
            panel.set_title("Panel description")
        plot.add_caption("Main figure title.")
        plt.show()
    """

    def __init__(self, fig_size: tuple = (12, 3)) -> None:
        """Initialize Plotter with default settings.

        Args:
            fig_size: Default figure size as (width, height) tuple
        """
        self.figure_number: int = 0
        self.fig_size: tuple[int, int] = fig_size
        self.font_size: int = 14
        self.fig = None
        self.axes = None
        self.panel_descriptions: list[str] = []  # Store panel descriptions

        plt.rcParams["font.family"] = "serif"
        plt.rcParams.update({"axes.titlesize": self.font_size})
        plt.rcParams.update({"axes.labelsize": self.font_size})
        plt.rcParams.update({"xtick.labelsize": self.font_size * 0.83})
        plt.rcParams.update({"ytick.labelsize": self.font_size * 0.83})
        plt.rcParams.update({"legend.fontsize": self.font_size * 0.83})

        # Configure tick marks to be inside and on all four sides
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.top"] = True
        plt.rcParams["xtick.bottom"] = True
        plt.rcParams["ytick.left"] = True
        plt.rcParams["ytick.right"] = True

    @property
    def panels(self):
        """
        Access the axes/panels created by new() or subplots().

        Returns an iterable of axes objects that works for both single and multiple subplots.
        Use in loops: `for panel in plot.panels: ...`

        Returns:
            Flattened array of axes objects, or None if no axes exist
        """
        if self.axes is None:
            return None
        # Handle both single axes and arrays of axes
        axes_array = np.atleast_1d(self.axes)
        if axes_array.ndim == 0:
            # Single axes object
            return np.array([self.axes])
        return axes_array.flat

    def new(
        self,
        fig_size=None,
        constrained_layout=True,
        ncols=1,
        nrows=1,
        fig_num=None,
        **kwargs,
    ):
        """
        Create a new figure with optional subplots.

        This method replaces the need to call plt.figure() or plt.subplots() directly.
        It automatically sets up the figure with consistent formatting and stores
        references for use by other Plotter methods.

        Args:
            fig_size: Figure size as (width, height) tuple. If None, uses default.
            constrained_layout: If True (default), automatically adjusts spacing to
                prevent overlap and maintain consistent plot areas regardless of labels.
            nrows: Number of subplot rows. Default is 1.
            ncols: Number of subplot columns. Default is 1.
            fig_num: Optional figure number. If None, auto-increments from current counter.
            **kwargs: Additional arguments passed to plt.subplots() (e.g., sharex, sharey)

        Returns:
            fig: The created matplotlib Figure object (also stored in self.fig)

        Example:
            plot.new(nrows=2, ncols=2, fig_size=(12, 8))
            plot.new(fig_num=5)  # Explicitly set figure number to 5
        """
        if fig_size is not None:
            self.fig_size = fig_size

        # Set figure number (if provided) or keep current counter
        # Note: figure_counter is incremented in add_caption(), not here
        if fig_num is not None:
            self.figure_number = fig_num
        else:
            self.figure_number += 1

        # Clear panel descriptions for new figure
        self.panel_descriptions = []

        # If nrows or ncols specified, create subplots
        if nrows is not None or ncols is not None:
            nrows = nrows or 1
            ncols = ncols or 1
            self.fig, self.axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=self.fig_size,
                constrained_layout=constrained_layout,
                **kwargs,
            )
        else:
            # Create simple figure
            self.fig = plt.figure(
                figsize=self.fig_size, constrained_layout=constrained_layout
            )
            self.axes = None

        return self.fig

    def subplots(
        self, nrows=1, ncols=1, fig_size=None, constrained_layout=True, **kwargs
    ):
        """
        Create a new figure with subplots (explicit return values).

        This is an alternative to new() that explicitly returns both fig and axes,
        similar to plt.subplots(). Use this if you prefer to capture the return values.
        Otherwise, use new() and access via plot.fig and plot.panels.

        Args:
            nrows: Number of rows of subplots
            ncols: Number of columns of subplots
            fig_size: Figure size as (width, height) tuple. If None, uses default.
            constrained_layout: If True (default), automatically adjusts spacing.
            **kwargs: Additional arguments passed to plt.subplots() (e.g., sharex, sharey)

        Returns:
            tuple: (fig, axes) - Figure and axes objects

        Example:
            fig, axes = plot.subplots(2, 2, fig_size=(12, 8))
            for ax in axes.flat:
                ax.plot(x, y)
        """
        if fig_size is not None:
            self.fig_size = fig_size

        self.fig, self.axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=self.fig_size,
            constrained_layout=constrained_layout,
            **kwargs,
        )
        return self.fig, self.axes

    def add_panel_description(self, description: str) -> None:
        """
        Manually add a description for a panel.

        This is an alternative to using panel.set_title(). Descriptions stored here
        will be automatically included when add_caption() is called. However, using
        set_title() is now preferred as it's more intuitive.

        Args:
            description: Text description of the panel (e.g., "Data series 1")

        Example (legacy approach):
            plot.new(nrows=2, ncols=2)
            for i, panel in enumerate(plot.panels):
                panel.plot(x, y[i])
                plot.add_panel_description(f"Data series {i+1}")
            plot.add_caption("Main title.")

        Preferred approach:
            plot.new(nrows=2, ncols=2)
            for i, panel in enumerate(plot.panels):
                panel.plot(x, y[i])
                panel.set_title(f"Data series {i+1}")
            plot.add_caption("Main title.")  # Automatically extracts titles
        """

        self.panel_descriptions.append(description)

    def add_caption(
        self,
        title: str,
        fig_num: int | None = None,
        filename: str | None = None,
        panel_descriptions: list[str] | None = None,
        labels: list[str] | None = None,
        start: str = "A",
        use_panel_titles: bool = True,
        auto_label_panels: bool = True,
    ) -> None:
        """
        Add a formatted caption to the figure with automatic features.

        This is the main method for finalizing your figure. It automatically:
        1. Labels all panels (A, B, C...) if there are multiple subplots
        2. Extracts panel titles and moves them to the caption
        3. Formats everything in publication-ready style

        The caption format is: "Figure X) Main title. A) Panel 1 desc. B) Panel 2 desc."

        Args:
            title: Main caption text describing the overall figure
            fig_num: Figure number (auto-increments if None)
            filename: Optional filename to save figure (e.g., "results.png")
            panel_descriptions: Optional list to override extracted panel titles
            labels: Optional custom labels (e.g., ['i', 'ii', 'iii']) instead of A, B, C
            start: Starting letter for auto-generated labels (default 'A')
            use_panel_titles: If True (default), extracts panel titles and removes them
            auto_label_panels: If True (default), automatically labels panels for multi-panel figures

        Example (simple):
            plot.new(nrows=2, ncols=2)
            for panel in plot.panels:
                panel.plot(x, y)
                panel.set_title("Description")
            plot.add_caption("Comparison of methods.")

        Example (custom):
            plot.add_caption("Results.", labels=['i', 'ii', 'iii'],
                           use_panel_titles=False)

        Args:
            title: Main caption text
            fig_num: Figure number (auto-increments if None)
            filename: Optional filename to save figure
            panel_descriptions: Optional list of descriptions for each subplot panel
            labels: Optional custom labels for panels (defaults to A, B, C, ...)
            start: Starting letter for auto-generated labels (default 'A')
            use_panel_titles: If True, extract panel titles and include in caption, then remove them
            auto_label_panels: If True, automatically call label_subplots() for multi-panel figures
        """
        if fig_num is None:
            fig_num = self.figure_number

        # Automatically label subplots if there are multiple panels
        if auto_label_panels and self.axes is not None:
            axes_flat = np.atleast_1d(self.axes).flatten()
            if len(axes_flat) > 1:
                self.label_subplots(labels=labels, start=start)

        # Build the caption text
        caption_text = title

        # Extract panel titles from axes if use_panel_titles is True and axes exist
        if use_panel_titles and self.axes is not None and panel_descriptions is None:
            axes_flat = np.atleast_1d(self.axes).flatten()
            panel_descriptions = []
            for ax in axes_flat:
                panel_title = ax.get_title()
                if panel_title:  # Only add if title exists
                    panel_descriptions.append(panel_title)
                    ax.set_title("")  # Remove the title from the plot

            # Only use extracted descriptions if we found any
            if not panel_descriptions:
                panel_descriptions = None

        # Use stored panel descriptions if not explicitly provided and not extracted
        if panel_descriptions is None and len(self.panel_descriptions) > 0:
            panel_descriptions = self.panel_descriptions

        # Add panel descriptions if provided
        if panel_descriptions is not None:
            # Generate labels if not provided
            if labels is None:
                labels = [chr(ord(start) + i) for i in range(len(panel_descriptions))]

            # Append each panel description with its label
            for label, desc in zip(labels, panel_descriptions):
                # Remove trailing period if present, we'll add it consistently
                desc_clean = desc.rstrip(".")
                caption_text += f" {label}) {desc_clean}."

        formatted_caption = f"$\\bf{{Figure\\ {fig_num})}}$ {caption_text}"
        plt.suptitle(
            formatted_caption,
            x=0,
            y=-0.025,
            ha="left",
            fontsize=self.font_size,
            wrap=True,
        )

        if filename is not None:
            filename = filename.replace(".", f"_{fig_num}.")
            plt.savefig(filename, dpi=600, bbox_inches="tight")

        return

    def add_label(self, fig_label: str | None = None) -> None:
        """
        Add a panel label to the top-left of the current axes.

        Note: This is rarely needed directly as add_caption() with auto_label_panels=True
        will automatically label all panels. Use this only for manual control.

        Args:
            fig_label: Label text (default: 'A'). Can be any string like 'A', 'i', '1', etc.

        Example:
            plt.subplot(1, 2, 1)
            plt.plot(x, y)
            plot.add_label('A')
        """
        if fig_label is None:
            fig_label = "A"

        formatted_caption = f"{fig_label}) "
        ax = plt.gca()
        ax.text(
            -0.1,
            0.98,
            s=formatted_caption,
            transform=ax.transAxes,
            fontsize=self.font_size * 1.2,
            fontweight="bold",
            va="top",
            ha="left",
        )
        return

    def label_subplots(self, labels=None, start="A"):
        """
        Add labels to all subplots automatically.

        Note: This is automatically called by add_caption() when auto_label_panels=True.
        You only need to call this directly if you want labels without a caption, or
        if you're using custom labels with add_caption().

        Args:
            labels: List of custom labels (e.g., ['i', 'ii', 'iii']), or None for A, B, C...
            start: Starting letter for auto-generated labels (default: 'A')

        Example:
            plot.new(nrows=1, ncols=3)
            # ... create plots ...
            plot.label_subplots(labels=['i', 'ii', 'iii'])
        """
        if self.axes is None:
            print("Warning: No subplots found. Use plot.subplots() first.")
            return

        # Handle both single axes and array of axes
        axes_flat = np.atleast_1d(self.axes).flatten()

        if labels is None:
            # Auto-generate labels starting from 'start'
            start_ord = ord(start)
            labels = [chr(start_ord + i) for i in range(len(axes_flat))]

        for ax, label in zip(axes_flat, labels):
            formatted_label = f"$\\bf{{{label})}}$"
            ax.text(
                -0.1,
                1.1,
                s=formatted_label,
                transform=ax.transAxes,
                fontsize=self.font_size * 1.2,
                fontweight="bold",
                va="top",
                ha="right",
            )
        return

    def plot_sequence(
        self,
        fiber: "ChromatinFiber",
        occupancy: bool = True,
        dyads: bool | np.ndarray = True,
        orfs: bool = False,
        energy: bool = False,
        methylation: bool = False,
    ) -> None:
        """
        Plot a chromatin fiber with nucleosome occupancy and positions.

        This is a specialized method for visualizing ChromatinFiber objects
        from the chromatin simulation code.

        Args:
            fiber: ChromatinFiber object containing sequence and nucleosome data
            occupancy: If True, plot nucleosome occupancy as shaded region
            dyads: If True, plot dyad positions as vertical lines. Can also be array of positions.
            orfs: If True, show open reading frames
            energy: If True, show sequence-dependent energy landscape
            methylation: If True, show methylation sites
        """

        plt.figure(figsize=(12, 2))
        plt.xlabel("i (bp)")
        plt.ylabel("occupancy")
        plt.xlim(min(fiber.index), max(fiber.index))
        plt.ylim(-0.1, 1.1)
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
        plt.tight_layout()

    def save_figure(self, filename: str = "figure") -> None:
        """
        Save the current figure to the figures/ directory.

        Note: add_caption() can also save figures if you pass the filename parameter.
        This method is for saving without adding a caption.

        Args:
            filename: Base filename (without path). Will be saved as
                     "figures/{filename}_{figure_counter}.png"

        Example:
            plot.save_figure("results")  # Saves to figures/results_1.png
        """
        fileout = f"figures/{filename}_{self.figure_number}.png"
        plt.savefig(fileout, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {fileout}")
