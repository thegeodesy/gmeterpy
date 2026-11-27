#!/usr/bin/env python

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import gmeterpy.units as u
from gmeterpy.models.vgg.floating import FloatingGravityFit

from .report import write_report


def main(argv=None):
    """Fit vertical gravity ties and write the station outputs."""
    parser = argparse.ArgumentParser(
        description="Polynomial fit of the vertical gravity gradients data"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="CSV with runn, level_1/level_2 in mm, delta_g in microgals; '-' for stdin",
        metavar="FILE",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Existing output directory (default: current directory)",
    )
    parser.add_argument(
        "--h-ref",
        type=float,
        default=0.710,
        metavar="METRES",
        help="Uncertainty reference height in metres (default: 0.710)",
    )
    parser.add_argument(
        "--subtract-gradient",
        type=float,
        metavar="GRADIENT",
        help="Gradient to subtract in microgals/m (default: automatic secant)",
    )
    parser.add_argument(
        "--highlight-set",
        metavar="ID",
        help="Highlight one measurement set and mute the others",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=(-10, 10),
        metavar=("MIN", "MAX"),
        help="Detrended gravity limits in microgals (default: -10 10)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=(0, 1.4),
        metavar=("MIN", "MAX"),
        help="Height axis limits in metres (default: 0 1.4)",
    )
    parser.add_argument(
        "--auto-limits",
        action="store_true",
        help="Use automatic limits for both axes",
    )
    parser.add_argument(
        "--height-range",
        type=float,
        nargs=2,
        default=(0.0001, 1.4),
        metavar=("MIN", "MAX"),
        help="Curve sampling range in metres (default: 0.0001 1.4)",
    )
    parser.add_argument(
        "-n", "--name", default="", type=str, help="Station name", metavar="STATION"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=int,
        default=2,
        help="Degree of the polynomial model (default: 2).",
        metavar="DEGREE",
    )
    parser.add_argument(
        "--leave-one-set-out",
        action="store_true",
        help="Report held-out tie errors by measurement set",
    )
    parser.add_argument(
        "--review-note",
        metavar="TEXT",
        help="Add a note to the report's inspection section when issues are flagged",
    )

    args = parser.parse_args(argv)
    try:
        source = (
            nullcontext(sys.stdin) if args.input == "-" else Path(args.input).open()
        )
        with source as stream:
            data = pd.read_csv(stream)
    except OSError as error:
        parser.error(str(error))

    fitter = FloatingGravityFit(degree=args.degree)
    ties = (
        data.level_1.to_numpy(dtype=float) / 1000 * u.m,
        data.level_2.to_numpy(dtype=float) / 1000 * u.m,
        data.delta_g.to_numpy(dtype=float) * u.uGal,
    )
    result = fitter.fit(*ties, set_ids=data.runn)
    held_out_ties = None
    if args.leave_one_set_out:
        try:
            _, held_out_ties = fitter.leave_one_set_out(*ties, set_ids=data.runn)
        except ValueError as error:
            parser.error(str(error))
    try:
        figure = result.plot(
            args.name,
            h_ref=args.h_ref,
            subtract_gradient=args.subtract_gradient,
            xlim=None if args.auto_limits else args.xlim,
            ylim=None if args.auto_limits else args.ylim,
            height_range=args.height_range,
            highlight_set=args.highlight_set,
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        figure.savefig(args.output_dir / f"{args.name}_all.png")
    finally:
        plt.close(figure)
    write_report(
        result,
        args.output_dir / f"vg_{args.name}_fit_all.txt",
        args.name,
        data=data,
        held_out_ties=held_out_ties,
        review_note=args.review_note,
    )
