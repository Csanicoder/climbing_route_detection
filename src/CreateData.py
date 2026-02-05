import argparse
import subprocess
from pathlib import Path
from subprocess import CalledProcessError

parser = argparse.ArgumentParser(
    description="Generate Hold, Analytics and Summary data in one."
)

parser.add_argument(
    "--route_name",
    type=str,
    help="Common name of route"
)

parser.add_argument(
    "--video_dir",
    type=Path,
    help="Path to directory output will go to"
)

parser.add_argument(
    "--data_dir",
    type=Path,
    help="Path to directory output will go to"
)

parser.add_argument(
    "--video_file",
    type=str,
    help="Ending of video file from the route name"
)

parser.add_argument(
    "--holds_file",
    type=str,
    help="Ending of holds file from the route name"
)

parser.add_argument(
    "--pose_file",
    type=str,
    help="Ending of pose file from the route name"
)

parser.add_argument(
    "--analytics_file",
    type=str,
    help="Ending of analytics file from the route name"
)

parser.add_argument(
    "--summary_file",
    type=str,
    help="Ending of summary file from the route name"
)

parser.add_argument(
    "--hold_generator",
    type=Path,
    help="Path to hold generator python file"
)

parser.add_argument(
    "--analytics_generator",
    type=Path,
    help="Path to analytics generator python file"
)

parser.add_argument(
    "--summary_generator",
    type=Path,
    help="Path to summary generator python file"
)

args = parser.parse_args()

# Run the hold generator script
subprocess.check_call(["python", args.hold_generator,
                 "--route_name", args.route_name,
                 "--video_dir", args.video_dir,
                 "--data_dir", args.data_dir,
                 "--video_file", args.video_file,
                 "--output", args.holds_file,
                 ])


# Run the frame analytics generator script
subprocess.check_call(["python", args.analytics_generator,
                 "--route_name", args.route_name,
                 "--data_dir", args.data_dir,
                 "--holds_file", args.holds_file,
                 "--pose_file", args.pose_file,
                 "--output", args.analytics_file
                 ])

# Run the summary generator script
subprocess.check_call(["python", args.summary_generator,
                 "--route_name", args.route_name,
                 "--data_dir", args.data_dir,
                 "--holds_file", args.holds_file,
                 "--pose_file", args.pose_file,
                 "--analytics_file", args.analytics_file,
                 "--output", args.summary_file
                 ])

