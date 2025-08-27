#!/usr/bin/env python

# ORB-SLAM Agent with Data Recording Capabilities - Circular Motion Version
# This agent combines ORB-SLAM localization with comprehensive data recording
# Designed to run in a circular pattern while collecting data every other frame
# Automatically stops recording when dataset reaches 5GB

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This agent demonstrates ORB-SLAM integration with comprehensive data recording.
It runs in a circular pattern while collecting sensor data, camera feeds, and pose estimates
every other frame until the dataset reaches 5GB, then automatically stops recording.
"""

from math import radians, cos, sin
import numpy as np
import pytransform3d.rotations as pyrot
from collections import defaultdict
import os
import time
import csv

import carla
from pytransform3d.transformations import concat
import orbslam3
import copy
from pynput import keyboard
import cv2 as cv

from maple.boulder import BoulderDetector
from maple.navigation import Navigator
from maple.pose import InertialApriltagEstimator
from maple.utils import *
from maple.pose.stereoslam import SimpleStereoSLAM
from maple.surface.map import SurfaceHeight, sample_surface, sample_lander
from maple.utils import carla_to_pytransform
# Import orbslam_utils inside the setup method to avoid circular import

# Import the data recording functionality
from lac_data import Recorder

""" Import the AutonomousAgent from the Leaderboard. """

# Import will be done inside the class to avoid circular import issues

""" Define the entry point so that the Leaderboard can instantiate the agent class. """


def get_entry_point():
    return "ORBSLAMRecorderAgentCircle"


""" Inherit the AutonomousAgent class. """

# Import AutonomousAgent locally to avoid circular imports
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "leaderboard"))
from leaderboard.autoagents.autonomous_agent import AutonomousAgent


class ORBSLAMRecorderAgentCircle(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Initialize the agent with ORB-SLAM and data recording capabilities for circular motion."""

        # Import only the specific functions we need to avoid circular import
        # from maple.pose.orbslam_utils import correct_pose_orientation, rotate_pose_in_place

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        # Basic agent attributes
        self.current_v = 0
        self.current_w = 0
        self.frame = 1

        # Camera configuration
        self._width = 1280
        self._height = 720
        self._active_side_cameras = False
        self._active_side_front_cameras = True

        # Data collection parameters
        self.recording_active = True
        self.recording_frequency = 2  # Record every other frame
        self.max_dataset_size_gb = 5  # Stop recording at 5GB

        # Initialize data recording
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.recorder = Recorder(
            self, f"orbslam_circular_{timestamp}.lac", self.max_dataset_size_gb
        )
        self.recorder.description(
            f"ORB-SLAM circular motion data collection - {timestamp}"
        )

        print("Data recording initialized successfully")
        print(f"Max dataset size: {self.max_dataset_size_gb}GB")
        print(f"Recording frequency: Every {self.recording_frequency} frames")

        # Initialize the sample list
        self.sample_list = []
        self.ground_truth_sample_list = []

        self.good_loc = True

        # Store previous boulder detections
        self.previous_detections = []

        self.navigator = Navigator(self)
        self.detector = BoulderDetector(
            self, carla.SensorPosition.FrontLeft, carla.SensorPosition.FrontRight
        )
        self.detectorBack = BoulderDetector(
            self, carla.SensorPosition.BackLeft, carla.SensorPosition.BackRight
        )

        self.g_map_testing = self.get_geometric_map()
        self.map_length_testing = self.g_map_testing.get_cell_number()

        for i in range(self.map_length_testing):
            for j in range(self.map_length_testing):
                self.g_map_testing.set_cell_height(i, j, 0)
                self.g_map_testing.set_cell_rock(i, j, 0)

        self.all_boulder_detections = []
        self.large_boulder_detections = [(0, 0, 2.5)]

        self.sample_list.extend(sample_lander(self))

        # Add position tracking for stuck detection
        self.position_history = []
        self.is_stuck = False
        self.unstuck_phase = 0
        self.unstuck_counter = 0

        # Tiered stuck detection parameters
        self.SEVERE_STUCK_FRAMES = 700
        self.SEVERE_STUCK_THRESHOLD = 0.4  # If moved less than 0.5m in 500 frames

        self.MILD_STUCK_FRAMES = 2000
        self.MILD_STUCK_THRESHOLD = 3.0  # If moved less than 3m in 1000 frames

        self.UNSTUCK_DISTANCE_THRESHOLD = (
            3.0  # How far to move to be considered unstuck
        )

        self.unstuck_sequence = [
            {"lin_vel": -0.45, "ang_vel": 0, "frames": 100},  # Backward
            {"lin_vel": 0, "ang_vel": 4, "frames": 60},  # Rotate clockwise
            {"lin_vel": 0.45, "ang_vel": 0, "frames": 150},  # Forward
            {"lin_vel": 0, "ang_vel": -4, "frames": 60},  # Rotate counter-clockwise
        ]

        # Add these variables for goal timeout tracking
        self.frames_since_goal_change = 0
        self.goal_timeout_threshold = 1000
        self.goal_timeout_active = False
        self.goal_timeout_counter = 0
        self.goal_timeout_duration = 200
        self.max_linear_velocity = 0.6  # Maximum linear velocity for timeout maneuver
        self.current_goal_index = 0  # Track which goal we're headed to

        # ORB-SLAM initialization (made optional)
        print("Starting ORB-SLAM initialization...")
        self.orb_vocab = (
            "/home/annikat/ORB-SLAM3-python/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt"
        )
        self.orb_cams_config = "/home/annikat/ORB-SLAM3-python/third_party/ORB_SLAM3/Examples/Stereo/LAC_cam.yaml"

        print(f"ORB vocabulary path: {self.orb_vocab}")
        print(f"ORB config path: {self.orb_cams_config}")

        # Check if ORB-SLAM files exist, if not use default paths
        if not os.path.exists(self.orb_vocab):
            print(f"ORB vocabulary not found at {self.orb_vocab}, using default")
            self.orb_vocab = "resources/ORBvoc.txt"
        if not os.path.exists(self.orb_cams_config):
            print(f"ORB config not found at {self.orb_cams_config}, using default")
            self.orb_cams_config = "resources/orbslam_config.yaml"

        print("Creating SimpleStereoSLAM instance...")

        # Skip ORB-SLAM initialization for now to avoid hanging
        print("Skipping ORB-SLAM initialization to avoid hanging...")
        self.orbslam = None
        print("ORB-SLAM disabled - will use simulated positioning")

        # Uncomment this section when ORB-SLAM is working properly
        # try:
        #     self.orbslam = SimpleStereoSLAM(self.orb_vocab, self.orb_cams_config)
        #     print("ORB-SLAM initialized successfully")
        # except Exception as e:
        #     print(f"ORB-SLAM initialization failed: {e}")
        #     print("Continuing without ORB-SLAM...")
        #     self.orbslam = None

        print("Setting up pose tracking...")
        self.positions = []

        print("Getting initial position...")
        try:
            self.init_pose = carla_to_pytransform(self.get_initial_position())
            print(f"Initial pose: {self.init_pose}")
        except Exception as e:
            print(f"Could not get initial position: {e}")
            print("Using simulated initial position...")
            # Create a simulated identity transform
            self.init_pose = np.eye(4)
            self.init_pose[0, 3] = 0.0  # x = 0
            self.init_pose[1, 3] = 0.0  # y = 0
            self.init_pose[2, 3] = 0.0  # z = 0
            print(f"Simulated initial pose: {self.init_pose}")

        self.prev_pose = None
        self.T_orb_to_global = None

        # Circular motion parameters
        self.circular_radius = 2.0  # meters - radius of the circle
        self.circular_velocity = 0.3  # m/s - conservative speed for data collection
        self.circular_angular_velocity = self.circular_velocity / self.circular_radius  # rad/s
        self.mission_duration = 700  
        self.start_time = None
        self.circle_center = np.array([0.0, 0.0])  # Center of the circle
        self.angle_offset = 0.0  # Starting angle offset

        print("ORB-SLAM Recorder Agent (Circular) initialized successfully!")
        print(f"Mission duration: {self.mission_duration} seconds ({self.mission_duration/60:.1f} minutes)")
        print(f"Circle radius: {self.circular_radius} meters")
        print(f"Circular velocity: {self.circular_velocity} m/s")
        print(f"Angular velocity: {self.circular_angular_velocity:.3f} rad/s")
        print("Setup method completed - agent is ready for circular motion")

    def check_if_stuck(self, current_position):
        """
        Check if the rover is stuck using a tiered approach:
        1. Severe stuck: very little movement in a short period
        2. Mild stuck: limited movement over a longer period
        Returns True if stuck, False otherwise.

        Only performs the check every 10 frames to improve performance.
        """
        if current_position is None:
            return False

        # Add current position to history
        self.position_history.append(current_position)

        # Keep only enough positions for the longer threshold check
        if len(self.position_history) > self.MILD_STUCK_FRAMES:
            self.position_history.pop(0)

        # Only perform stuck detection every 10 frames to improve performance
        if self.frame % 10 != 0:
            return False

        # Check for severe stuck condition (shorter timeframe)
        if len(self.position_history) >= self.SEVERE_STUCK_FRAMES:
            severe_check_position = self.position_history[-self.SEVERE_STUCK_FRAMES]
            dx = current_position[0] - severe_check_position[0]
            dy = current_position[1] - severe_check_position[1]
            severe_distance_moved = np.sqrt(dx**2 + dy**2)

            if severe_distance_moved < self.SEVERE_STUCK_THRESHOLD:
                print(
                    f"SEVERE STUCK DETECTED! Moved only {severe_distance_moved:.2f}m in the last {self.SEVERE_STUCK_FRAMES} frames."
                )
                return True

        # Check for mild stuck condition (longer timeframe)
        if len(self.position_history) >= self.MILD_STUCK_FRAMES:
            mild_check_position = self.position_history[0]  # Oldest position
            dx = current_position[0] - mild_check_position[0]
            dy = current_position[1] - mild_check_position[1]
            mild_distance_moved = np.sqrt(dx**2 + dy**2)

            if mild_distance_moved < self.MILD_STUCK_THRESHOLD:
                print(
                    f"MILD STUCK DETECTED! Moved only {mild_distance_moved:.2f}m in the last {self.MILD_STUCK_FRAMES} frames."
                )
                return True

        return False

    def get_unstuck_control(self):
        # Same as before - no changes needed here
        current_phase = self.unstuck_sequence[self.unstuck_phase]
        lin_vel = current_phase["lin_vel"]
        ang_vel = current_phase["ang_vel"]
        self.unstuck_counter += 1

        if self.unstuck_counter >= current_phase["frames"]:
            self.unstuck_phase = (self.unstuck_phase + 1) % len(self.unstuck_sequence)
            self.unstuck_counter = 0
            print(f"Moving to unstuck phase {self.unstuck_phase}")

        return lin_vel, ang_vel

    def use_fiducials(self):
        """We want to use the fiducials, so we return True."""
        return False

    def sensors(self):
        """In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048)
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light."""

        sensors = {
            carla.SensorPosition.Front: {
                "camera_active": False,
                "light_intensity": 0.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.FrontLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.Left: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._width}",
                "use_semantic": False,
            },
            carla.SensorPosition.Right: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.BackLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.BackRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
            carla.SensorPosition.Back: {
                "camera_active": False,
                "light_intensity": 0.0,
                "width": f"{self._width}",
                "height": f"{self._height}",
                "use_semantic": False,
            },
        }
        return sensors

    def run_step(self, input_data):
        """Wraps MAPLE logic incase something goes wrong."""

        print(
            f"RUN_STEP CALLED! Frame {self.frame}, input_data keys: {list(input_data.keys()) if input_data else 'None'}"
        )

        try:
            print(f"Frame {self.frame}: Starting run_step")
            result = self.run_step_unsafe(input_data)
            print(f"Frame {self.frame}: run_step completed successfully")
            return result
        except Exception as e:
            print(f"FATAL ERROR in frame {self.frame}: {e}")
            import traceback

            traceback.print_exc()
            self.finalize()
            self.mission_complete()

    def run_step_unsafe(self, input_data):
        """Execute one step of the circular navigation with data recording."""

        # Initialize start time on first frame
        if self.frame == 1:
            self.start_time = time.time()
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))
            print("Starting circular motion data collection mission...")

        # Check mission duration
        if self.start_time and (time.time() - self.start_time) > self.mission_duration:
            print(
                f"Mission duration reached ({self.mission_duration}s) - completing mission"
            )
            self.finalize()
            self.mission_complete()
            return carla.VehicleVelocityControl(0, 0)

        # Check if recording should stop due to file size
        if self.recording_active and self.recorder.is_done():
            print(
                f"Dataset size limit reached ({self.max_dataset_size_gb}GB) - stopping recording"
            )
            self.recording_active = False
            self.recorder.stop()

        # Data recording (every other frame) - work around transform issues
        if self.recording_active and self.frame % self.recording_frequency == 0:
            try:
                # Try to get real transform first
                transform = self.get_transform()
                if transform is not None and transform.location is not None:
                    print(
                        f"Frame {self.frame}: Starting data recording with real transform..."
                    )
                    self.recorder(self.frame, input_data)
                    print(f"Frame {self.frame}: Data recording completed")
                else:
                    # Use simulated transform for recording
                    print(
                        f"Frame {self.frame}: Using simulated transform for recording..."
                    )
                    # Create a simulated transform that moves in a circle
                    # simulated_x = self.frame * 0.01  # Move 1cm per frame
                    # simulated_y = 0.0
                    # simulated_z = 0.0

                    # Try to record with simulated data
                    try:
                        self.recorder(self.frame, input_data)
                        print(
                            f"Frame {self.frame}: Data recording completed (simulated)"
                        )
                    except Exception as recording_error:
                        print(
                            f"Frame {self.frame}: Recording failed even with simulation: {recording_error}"
                        )

                if self.frame % 100 == 0:  # Log every 100 frames
                    print(f"Recording frame {self.frame}")
            except Exception as e:
                print(f"Recording error in frame {self.frame}: {e}")
                import traceback

                traceback.print_exc()

        # ORB-SLAM processing (every frame for localization)
        print(f"Frame {self.frame}: Checking sensor data...")

        # Validate sensor data exists
        if "Grayscale" not in input_data:
            print(f"Frame {self.frame}: No Grayscale data in input_data")
            estimate = self.init_pose
            self.prev_pose = estimate
            return carla.VehicleVelocityControl(0, 0)

        sensor_data_frontleft = input_data["Grayscale"].get(
            carla.SensorPosition.FrontLeft
        )
        sensor_data_frontright = input_data["Grayscale"].get(
            carla.SensorPosition.FrontRight
        )

        print(
            f"Frame {self.frame}: FrontLeft sensor: {'OK' if sensor_data_frontleft is not None else 'FAIL'}, FrontRight sensor: {'OK' if sensor_data_frontright is not None else 'FAIL'}"
        )

        if self.frame < 50:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))
            estimate = self.init_pose
            self.prev_pose = estimate

        elif (
            sensor_data_frontleft is not None
            and sensor_data_frontright is not None
            and self.frame >= 50
        ):
            # ORB-SLAM is disabled for now, use simulated circular pose
            print(f"Frame {self.frame}: ORB-SLAM disabled, using simulated circular pose")
            estimate = self.init_pose.copy()
            
            # Calculate circular motion
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            angle = self.circular_angular_velocity * elapsed_time + self.angle_offset
            
            # Circular path: x = r*cos(θ), y = r*sin(θ)
            estimate[0, 3] = self.circle_center[0] + self.circular_radius * cos(angle)
            estimate[1, 3] = self.circle_center[1] + self.circular_radius * sin(angle)
            estimate[2, 3] = 0.0  # Stay on ground level

        elif (
            sensor_data_frontleft is None
            and sensor_data_frontright is None
            and self.frame >= 50
        ):
            estimate = self.prev_pose

        # Store the current pose for next iteration
        self.prev_pose = estimate

        # Get a position estimate for the rover
        roll, pitch, yaw = pyrot.euler_from_matrix(
            estimate[:3, :3], i=0, j=1, k=2, extrinsic=True
        )
        if np.abs(pitch) > np.deg2rad(80) or np.abs(roll) > np.deg2rad(80):
            self.set_front_arm_angle(radians(0))
            self.set_back_arm_angle(radians(0))
        else:
            self.set_front_arm_angle(radians(60))

        current_position = (
            (estimate[0, 3], estimate[1, 3]) if estimate is not None else None
        )

        if current_position is not None:
            # Always update position history
            self.position_history.append(current_position)

            # Keep only enough positions for the longer threshold check
            if len(self.position_history) > self.MILD_STUCK_FRAMES:
                self.position_history.pop(0)

            # Only check if stuck every 10 frames for performance
            if not self.is_stuck and self.frame % 10 == 0:
                self.is_stuck = self.check_if_stuck(current_position)
            elif self.is_stuck:
                # Check if we've moved enough to consider ourselves unstuck
                if len(self.position_history) > 0:
                    old_position = self.position_history[0]
                    dx = current_position[0] - old_position[0]
                    dy = current_position[1] - old_position[1]
                    distance_moved = np.sqrt(dx**2 + dy**2)

                    if distance_moved > self.UNSTUCK_DISTANCE_THRESHOLD:
                        print(
                            f"UNSTUCK! Moved {distance_moved:.2f}m - resuming normal operation."
                        )
                        self.is_stuck = False
                        self.unstuck_phase = 0
                        self.unstuck_counter = 0
                        # Clear position history to reset stuck detection
                        self.position_history = []

        # Boulder detection and obstacle avoidance (every 20 frames)
        if self.frame % 20 == 0:
            try:
                detections, _ = self.detector(input_data)
                detections_back, _ = self.detectorBack(input_data)

                large_boulders_detections = self.detector.get_large_boulders()

                # Get all detections in the world frame
                rover_world = estimate
                boulders_world = [
                    concat(boulder_rover, rover_world) for boulder_rover in detections
                ]

                boulders_world_back = [
                    concat(boulder_rover, rover_world)
                    for boulder_rover in detections_back
                ]

                large_boulders_detections = [
                    concat(boulder_rover, rover_world)
                    for boulder_rover in large_boulders_detections
                ]

                large_boulders_xyr = [
                    (b_w[0, 3], b_w[1, 3], 0.25) for b_w in large_boulders_detections
                ]

                # Now pass the (x, y, r) tuples to your navigator or wherever they need to go
                self.navigator.add_large_boulder_detection(large_boulders_xyr)
                self.large_boulder_detections.extend(large_boulders_xyr)

                # If you just want X, Y coordinates as a tuple
                boulders_xy = [(b_w[0, 3], b_w[1, 3]) for b_w in boulders_world]
                boulders_xy_back = [
                    (b_w[0, 3], b_w[1, 3]) for b_w in boulders_world_back
                ]

                self.all_boulder_detections.extend(boulders_xy)
                print("len(boulders)", len(self.all_boulder_detections))
                self.all_boulder_detections.extend(boulders_xy_back)

            except Exception as e:
                print(f"Error processing detections: {e}")
                print(f"Error details: {str(e)}")

        # Navigation control - circular motion
        if self.is_stuck:
            # Execute unstuck sequence
            goal_lin_vel, goal_ang_vel = self.get_unstuck_control()
            print(
                f"Unstuck maneuver: lin_vel={goal_lin_vel}, ang_vel={goal_ang_vel}"
            )
        else:
            # Circular motion navigation
            goal_lin_vel = self.circular_velocity
            goal_ang_vel = self.circular_angular_velocity

        # Surface sampling when stopped
        if goal_lin_vel == 0 and self.frame % 20 == 0:
            self.sample_list.extend(sample_surface(estimate, 60))

        # Progress logging
        if self.frame % 100 == 0:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            print(
                f"Frame {self.frame} - Elapsed: {elapsed_time:.1f}s - Recording: {'ON' if self.recording_active else 'OFF'}"
            )

        self.frame += 1

        print("Frame: ", self.frame)

        if self.frame < 50:
            goal_lin_vel, goal_ang_vel = 0.0, 0.0

        print("goal lin vel: ", goal_lin_vel)
        print("goal ang vel: ", goal_ang_vel)

        # Finally, apply the resulting velocities
        control = carla.VehicleVelocityControl(goal_lin_vel, goal_ang_vel)

        return control

    def finalize(self):
        """Clean up and finalize data collection."""
        print("Finalizing ORB-SLAM Recorder Agent (Circular)...")

        # Stop recording if still active
        if self.recording_active:
            print("Stopping data recording...")
            self.recorder.stop()
            self.recording_active = False

        min_det_threshold = 2

        if self.frame > 15000:
            min_det_threshold = 2

        if self.frame > 35000:
            min_det_threshold = 3

        g_map = self.get_geometric_map()
        gt_map_array = g_map.get_map_array()

        N = gt_map_array.shape[
            0
        ]  # should be 179 if you are spanning -13.425 to 13.425 by 0.15
        x_min, y_min = gt_map_array[0][0][0], gt_map_array[0][0][0]
        resolution = 0.15

        # Calculate indices for center 2x2m region
        center_x_min_idx = int(round((-1 - x_min) / resolution))  # -.5m in x
        center_x_max_idx = int(round((1 - x_min) / resolution))  # +.5m in x
        center_y_min_idx = int(round((-1 - y_min) / resolution))  # -.5m in y
        center_y_max_idx = int(round((1 - y_min) / resolution))  # +.5m in y

        # setting all rock locations to 0
        for i in range(self.map_length_testing):
            for j in range(self.map_length_testing):
                self.g_map_testing.set_cell_rock(i, j, 0)

        clusters = defaultdict(list)
        filtered_detections = []

        # First pass: create clusters
        for x_rock, y_rock in self.all_boulder_detections:
            # Convert to grid coordinates
            i = int(round((x_rock - x_min) / resolution))
            j = int(round((y_rock - y_min) / resolution))

            # Create cluster key based on grid cell
            cluster_key = (i, j)
            clusters[cluster_key].append([x_rock, y_rock])

        final_clusters = []

        # Second pass: process clusters and filter outliers
        for (i, j), detections in clusters.items():
            # Skip clusters with less than 2 detections
            if len(detections) < min_det_threshold:
                continue

            final_clusters.extend(clusters[(i, j)])

            # Skip if in center region
            if (
                center_x_min_idx <= i <= center_x_max_idx
                and center_y_min_idx <= j <= center_y_max_idx
            ):
                continue

            # Sanity check: make sure we are within bounds
            if 0 <= i < N and 0 <= j < N:
                # Calculate cluster center
                x_center = float(np.mean([x for x, y in detections]))
                y_center = float(np.mean([y for x, y in detections]))

                # Convert back to grid coordinates for the map
                i_center = int(round((x_center - x_min) / resolution))
                j_center = int(round((y_center - y_min) / resolution))

                # Set rock location at cluster center
                self.g_map_testing.set_cell_rock(i_center, j_center, 1)

                # Store the cluster center as a simple list
                filtered_detections.append([x_center, y_center])

        # Initialize the data class to get estimates for all the squares
        surfaceHeight = SurfaceHeight(g_map)

        # Generate the actual map with the sample list
        if len(self.sample_list) > 0:
            surfaceHeight.set_map(self.sample_list)
            print(f"Surface map finalized with {len(self.sample_list)} samples")

        # Process boulder detections
        if len(self.all_boulder_detections) > 0:
            print(f"Final boulder detections: {len(self.all_boulder_detections)}")

        print("ORB-SLAM Recorder Agent (Circular) finalized successfully!")

    def on_press(self, key):
        """This is the callback executed when a key is pressed. If the key pressed is either the up or down arrow, this method will add
        or subtract target linear velocity. If the key pressed is either the left or right arrow, this method will set a target angular
        velocity of 0.6 radians per second."""

        if key == keyboard.Key.up:
            self.current_v += 0.1
            self.current_v = np.clip(self.current_v, 0, 0.3)
        if key == keyboard.Key.down:
            self.current_v -= 0.1
            self.current_v = np.clip(self.current_v, -0.3, 0)
        if key == keyboard.Key.left:
            self.current_w = 0.6
        if key == keyboard.Key.right:
            self.current_w = -0.6

    def on_release(self, key):
        """This method sets the angular or linear velocity to zero when the arrow key is released. Stopping the robot."""

        if key == keyboard.Key.up:
            self.current_v = 0
        if key == keyboard.Key.down:
            self.current_v = 0
        if key == keyboard.Key.left:
            self.current_w = 0
        if key == keyboard.Key.right:
            self.current_w = 0

        """ Press escape to end the mission. """
        if key == keyboard.Key.esc:
            print("Manual mission termination requested")
            self.finalize()
            self.mission_complete()
            cv.destroyAllWindows() 