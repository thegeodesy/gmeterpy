#!/usr/bin/env python

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import gmeterpy.units as u
from gmeterpy.models.vgg.floating import FloatingGravityFit

from .input import LEGACY_COLUMNS, numeric_column, resolve_columns
from .report import write_report


def main(argv=None):
    """Fit vertical gravity ties and write the station outputs."""
    parser = argparse.ArgumentParser(
        description="Polynomial fit of the vertical gravity gradients data"
    )
    files = parser.add_argument_group("Input and output")
    files.add_argument(
        "file",
        nargs="?",
        help=(
            "Tie CSV with set_id (shared adjusted levels), h1, h2, and delta_g. "
            "Use '-' for stdin."
        ),
        metavar="FILE",
    )
    files.add_argument(
        "-i",
        "--input",
        help="Alternative to positional FILE. Use '-' for stdin.",
        metavar="FILE",
    )
    files.add_argument(
        "--columns",
        action="extend",
        nargs="+",
        default=[],
        metavar="FIELD=COLUMN",
        help="Map set_id, h1, h2, or delta_g to CSV headers",
    )
    files.add_argument(
        "--height-unit",
        default="mm",
        metavar="UNIT",
        help="Input unit for both heights (default: mm)",
    )
    files.add_argument(
        "--gravity-unit",
        default="uGal",
        metavar="UNIT",
        help="Input unit for gravity differences (default: uGal)",
    )

    files.add_argument(
        "-n",
        "--name",
        help="Station label for the plot and report. Also prefixes filenames for stdin.",
        metavar="STATION",
    )
    files.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Existing output directory (default: current directory)",
    )
    files.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files",
    )

    fitting = parser.add_argument_group("Fitting")
    fitting.add_argument(
        "-d",
        "--degree",
        type=int,
        default=2,
        help="Degree of the polynomial model (default: 2).",
        metavar="DEGREE",
    )
    fitting.add_argument(
        "--exclude-set",
        action="append",
        default=[],
        metavar="ID",
        help="Exclude a measurement set from the fit; repeat for multiple sets",
    )
    fitting.add_argument(
        "--auto-exclude",
        action="store_true",
        help="Refit once after excluding sets with flagged observations",
    )

    reporting = parser.add_argument_group("Diagnostics and reporting")
    reporting.add_argument(
        "--leave-one-set-out",
        action="store_true",
        help="Report held-out tie errors by measurement set",
    )
    reporting.add_argument(
        "--review-note",
        metavar="TEXT",
        help="Record a review note in the report",
    )

    plotting = parser.add_argument_group("Plotting")
    plotting.add_argument(
        "--no-plot",
        action="store_true",
        help="Write the report without creating a plot",
    )
    plotting.add_argument(
        "--highlight-set",
        metavar="ID",
        help="Highlight one measurement set in the plot and mute the others",
    )
    plotting.add_argument(
        "--h-ref",
        type=float,
        default=0.710,
        metavar="METRES",
        help="Reference height for the plot's uncertainty band in metres (default: 0.710)",
    )
    plotting.add_argument(
        "--subtract-gradient",
        type=float,
        metavar="GRADIENT",
        help="Gradient subtracted in the plot in microgals/m (default: automatic secant)",
    )
    plotting.add_argument(
        "--auto-limits",
        action="store_true",
        help="Use automatic limits for both plot axes",
    )
    plotting.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=(-10, 10),
        metavar=("MIN", "MAX"),
        help="Plot limits for detrended gravity in microgals (default: -10 10)",
    )
    plotting.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=(0, 1.4),
        metavar=("MIN", "MAX"),
        help="Height limits for the plot axis and fitted curve in metres (default: 0 1.4)",
    )

    args = parser.parse_args(argv)
    if args.file is not None and args.input is not None:
        parser.error("Supply the input as FILE or with --input, not both")
    input_file = args.file if args.file is not None else args.input
    if input_file is None:
        parser.error("Supply an input FILE, or use '-' for stdin")
    station = args.name or ""
    input_path = None if input_file == "-" else Path(input_file)
    source_stem = station if input_path is None else input_path.stem
    output_stem = f"{source_stem}_vgg_fit" if source_stem else "vgg_fit"
    report_path = args.output_dir / f"{output_stem}.txt"
    plot_path = None if args.no_plot else args.output_dir / f"{output_stem}.png"
    output_paths = [report_path] if plot_path is None else [plot_path, report_path]
    if input_path is not None:
        source_path = input_path.resolve()
        if any(path.resolve() == source_path for path in output_paths):
            parser.error(f"An output path points to the input file {input_file}")
    existing = [path for path in output_paths if path.exists() or path.is_symlink()]
    if existing and not args.overwrite:
        parser.error(
            "Output already exists. Use --overwrite or a different output directory. "
            + ", ".join(map(str, existing))
        )
    for path in existing:
        if path.is_dir():
            parser.error(f"Output path is a directory: {path}")
    try:
        source = nullcontext(sys.stdin) if input_path is None else input_path.open()
        with source as stream:
            data = pd.read_csv(stream)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    held_out_ties = None
    try:
        fitter = FloatingGravityFit(degree=args.degree)
        if data.empty:
            raise ValueError("Input CSV contains no ties")
        overrides = {}
        for mapping in args.columns:
            field, separator, column = mapping.partition("=")
            if not separator or field not in LEGACY_COLUMNS or not column:
                raise ValueError(
                    f"Invalid column mapping {mapping!r}. Use set_id=, h1=, h2=, or delta_g= followed by a CSV header."
                )
            if field in overrides:
                raise ValueError(f"Multiple column mappings for {field}")
            overrides[field] = column
        columns = resolve_columns(data, overrides)
        for field, column in columns.items():
            if column not in data.columns:
                raise ValueError(f"Missing CSV column {column!r} for {field}")
        if len(set(columns.values())) != len(columns):
            raise ValueError("Each fit field must map to a different CSV column")

        height_unit = u.Unit(args.height_unit)
        gravity_unit = u.Unit(args.gravity_unit)
        if not height_unit.is_equivalent(u.m):
            raise ValueError("--height-unit must be a length unit")
        if not gravity_unit.is_equivalent(u.uGal):
            raise ValueError("--gravity-unit must be an acceleration unit")
        h1 = numeric_column(data, columns["h1"]) * height_unit
        h2 = numeric_column(data, columns["h2"]) * height_unit
        delta_g = numeric_column(data, columns["delta_g"]) * gravity_unit
        set_id = data[columns["set_id"]]
        missing = set_id.isna().to_numpy().nonzero()[0]
        if missing.size:
            raise ValueError(
                f"Column {columns['set_id']!r} is missing a set ID in data row {missing[0] + 1}."
            )
        ids_by_label = {str(identifier): identifier for identifier in set_id.unique()}
        exclusions = [ids_by_label.get(label, label) for label in args.exclude_set]
        result = fitter.fit(
            h1,
            h2,
            delta_g,
            set_id=set_id,
            exclude_sets=exclusions,
            auto_exclude=args.auto_exclude,
        )
        if args.leave_one_set_out:
            _, held_out_ties = fitter.leave_one_set_out(
                h1, h2, delta_g, set_id=set_id, exclude_sets=result.excluded_set_ids
            )
    except ValueError as error:
        parser.error(str(error))
    with TemporaryDirectory(dir=args.output_dir, prefix=".vgfit-") as directory:
        staged_report = Path(directory) / "report.txt"
        outputs = [(staged_report, report_path)]
        if plot_path is not None:
            import matplotlib.pyplot as plt

            height_limits = None if args.auto_limits else args.ylim
            try:
                figure = result.plot(
                    station,
                    h_ref=args.h_ref,
                    subtract_gradient=args.subtract_gradient,
                    xlim=None if args.auto_limits else args.xlim,
                    ylim=height_limits,
                    height_range=height_limits,
                    highlight_set=args.highlight_set,
                )
            except ValueError as error:
                parser.error(str(error))
            staged_plot = Path(directory) / "plot.png"
            try:
                figure.savefig(staged_plot)
            finally:
                plt.close(figure)
            outputs.insert(0, (staged_plot, plot_path))

        write_report(
            result,
            staged_report,
            station,
            data=data,
            input_source="stdin" if input_path is None else input_file,
            columns=columns,
            height_unit=height_unit,
            gravity_unit=gravity_unit,
            held_out_ties=held_out_ties,
            review_note=args.review_note,
        )
        try:
            _publish_outputs(outputs, overwrite=args.overwrite)
        except FileExistsError as error:
            parser.error(f"Output already exists: {error.filename}. Use --overwrite.")
    if plot_path is not None:
        print(f"Wrote plot to {plot_path}")
    print(f"Wrote report to {report_path}")


def _publish_outputs(outputs: list[tuple[Path, Path]], *, overwrite: bool) -> None:
    """Move completed files into place, reserving destinations when needed."""
    reserved = []
    try:
        if not overwrite:
            for _, path in outputs:
                path.touch(exist_ok=False)
                reserved.append(path)
        for staged, path in outputs:
            staged.replace(path)
    except OSError:
        for path in reserved:
            path.unlink(missing_ok=True)
        raise
