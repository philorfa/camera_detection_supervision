from argparse import ArgumentParser
from src.utils.vehicle_modification.vehicle_examination import examine_rgb_frames


if __name__ == "__main__":
    """
Main entry point for analyzing RGB frames for identifying vehicle characteristics.

This script processes RGB frames to detect multiple frames of cars and optionally visualize
the frames based on the command-line arguments provided.

Command Line Arguments:
    -v, --visualize (bool):
        Flag to indicate whether to display the frame when an update is
        detected. Default is True.
"""

    parser = ArgumentParser(add_help=False)
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    optional.add_argument(
        "-v",
        "--visualize",
        type=bool,
        default=True,
        help=("Show frame when update is detected"),
    )

    args = parser.parse_args()

    examine_rgb_frames(visualize=args.visualize)
