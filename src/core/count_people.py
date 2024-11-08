from argparse import ArgumentParser
from src.utils.thermal_counting.thermal_live_frame import (
    analyze_thermal_frames)

if __name__ == "__main__":
    """
Main entry point for analyzing thermal frames.

This script processes thermal frames to detect updates and optionally
visualize the frames based on the command-line arguments provided.

Command Line Arguments:
    -v, --visualize (bool):
        Flag to indicate whether to display the frame when an update is
        detected. Default is False.
"""

    parser = ArgumentParser(add_help=False)
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    optional.add_argument(
        "-vd",
        "--visualize_detection",
        type=bool,
        default=True,
        help=("Show frame when update is detected"),
    )

    optional.add_argument(
        "-vx",
        "--visualize_xai",
        type=bool,
        default=True,
        help=("Visualize XAI"),
    )

    args = parser.parse_args()

    analyze_thermal_frames(visualize_detection=args.visualize_detection,
                       visualize_xai=args.visualize_xai)
