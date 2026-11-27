#!/usr/bin/env python

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

import gmeterpy.units as u
from gmeterpy.models.vgg.floating import FloatingGravityFit


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
    detrend = parser.add_mutually_exclusive_group()
    detrend.add_argument(
        "--subtract-gradient",
        type=float,
        metavar="GRADIENT",
        help="Gradient to subtract in microgals/m (default: automatic secant)",
    )
    detrend.add_argument(
        "--no-detrend", action="store_true", help="Show the original polynomial"
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=(-10, 10),
        metavar=("MIN", "MAX"),
        help="Gravity axis limits in microgals (default: -10 10)",
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

    args = parser.parse_args(argv)
    try:
        source = (
            nullcontext(sys.stdin) if args.input == "-" else Path(args.input).open()
        )
        with source as stream:
            data = pd.read_csv(stream)
    except OSError as error:
        parser.error(str(error))

    result = FloatingGravityFit(degree=args.degree).fit(
        data.level_1.to_numpy(dtype=float) / 1000 * u.m,
        data.level_2.to_numpy(dtype=float) / 1000 * u.m,
        data.delta_g.to_numpy(dtype=float) * u.uGal,
        set_ids=data.runn,
    )
    gradient = args.subtract_gradient
    if args.no_detrend:
        gradient = None
    elif gradient is None:
        h_min = result.observations.h.min()
        model = result.fitted_model
        if h_min == 1.0:
            slope, _ = model.gradient(1.0 * u.m)
        else:
            slope, _ = model.mean_gradient(1.0 * u.m, h_min * u.m)
        gradient = slope.to_value(u.uGal / u.m)
    figure = result.plot(
        args.name,
        h_ref=args.h_ref,
        subtract_gradient=cast(float | None, gradient),
        xlim=None if args.auto_limits else args.xlim,
        ylim=None if args.auto_limits else args.ylim,
        height_range=args.height_range,
    )
    try:
        figure.savefig(args.output_dir / f"{args.name}_all.png")
    finally:
        plt.close(figure)
    result.report(args.output_dir / f"vg_{args.name}_fit_all.txt", args.name, data=data)
