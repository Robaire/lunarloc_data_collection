#!/usr/bin/env python

# ORB-SLAM Agent with Data Recording Capabilities
# This agent combines ORB-SLAM localization with comprehensive data recording
# Designed to run in a straight line while collecting data every other frame
# Automatically stops recording when dataset reaches 5GB

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This agent demonstrates ORB-SLAM integration with comprehensive data recording.
It runs in a straight line while collecting sensor data, camera feeds, and pose estimates
every other frame until the dataset reaches 5GB, then automatically stops recording.
"""

import time
from math import radians

import carla
from lac_data import Recorder

from leaderboard.autoagents.autonomous_agent import AutonomousAgent


def get_entry_point():
    return "StraightLineAgent"


class StraightLineAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        """Initialize the agent with ORB-SLAM and data recording capabilities."""

        # Import only the specific functions we need to avoid circular import
        # from maple.pose.orbslam_utils import correct_pose_orientation, rotate_pose_in_place

        # Configuration
        # TODO: It might be necessary to add randomness to the velocity and angular velocity
        # in order to differentiate between different runs of the same mission
        self.linear_velocity = 0.3  # m/s
        self.angular_velocity = 0.0  # rad/s
        self.target_distance = 20  # m
        self.mission_duration = 700  # s

        # Camera configuration
        self.width = 1280
        self.height = 720

        # Frame counter
        self.frame = 1

        # Data collection parameters
        self.max_dataset_size_gb = 5  # Stop recording at 5GB

        # Initialize data recording
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.recorder = Recorder(
            self, f"orbslam_straight_line_{timestamp}.lac", self.max_dataset_size_gb
        )
        self.recorder.description(
            f"ORB-SLAM straight line data collection - {timestamp}"
        )

        print("Data recording initialized successfully")
        print(f"Max dataset size: {self.max_dataset_size_gb}GB")

        print("ORB-SLAM Recorder Agent initialized successfully!")
        print(f"Mission duration: {self.mission_duration} seconds")
        print(f"Straight line velocity: {self.linear_velocity} m/s")
        print("Setup method completed - agent is ready for execution")

    def use_fiducials(self):
        return False

    def sensors(self):
        sensors = {
            carla.SensorPosition.Front: {
                "camera_active": False,
                "light_intensity": 0.0,
                "width": f"{self.width}",
                "height": f"{self.height}",
                "use_semantic": False,
            },
            carla.SensorPosition.FrontLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self.width}",
                "height": f"{self.height}",
                "use_semantic": False,
            },
            carla.SensorPosition.FrontRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self.width}",
                "height": f"{self.height}",
                "use_semantic": False,
            },
            carla.SensorPosition.Left: {
                "camera_active": False,
                "light_intensity": 0.0,
                "width": f"{self.width}",
                "height": f"{self.height}",
                "use_semantic": False,
            },
            carla.SensorPosition.Right: {
                "camera_active": False,
                "light_intensity": 0.0,
                "width": f"{self.width}",
                "height": f"{self.height}",
                "use_semantic": False,
            },
            carla.SensorPosition.BackLeft: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self.width}",
                "height": f"{self.height}",
                "use_semantic": False,
            },
            carla.SensorPosition.BackRight: {
                "camera_active": True,
                "light_intensity": 1.0,
                "width": f"{self.width}",
                "height": f"{self.height}",
                "use_semantic": False,
            },
            carla.SensorPosition.Back: {
                "camera_active": False,
                "light_intensity": 0.0,
                "width": f"{self.width}",
                "height": f"{self.height}",
                "use_semantic": False,
            },
        }
        return sensors

    def run_step(self, input_data):
        try:
            result = self.run_step_unsafe(input_data)
            return result
        except Exception as e:
            print(f"FATAL ERROR in frame {self.frame}: {e}")

            # import traceback # THIS SHOULDN'T BE IMPORTED HERE
            # traceback.print_exc()

            # self.finalize() # THIS GETS CALLED AUTOMATICALLY BY THE MISSION COMPLETE METHOD
            self.mission_complete()

    def run_step_unsafe(self, input_data):
        """Execute one step of the straight line navigation with data recording."""

        # Raise the arms
        if self.frame == 1:
            self.set_front_arm_angle(radians(60))
            self.set_back_arm_angle(radians(60))

        # Wait a little bit for the arms to get out of the way
        # TODO: Check if this is long enough
        if self.get_mission_time() < 10:
            return carla.VehicleVelocityControl(0, 0)

        # Check if the mission duration has elapsed
        if self.get_mission_time() > self.mission_duration + 10:
            self.mission_complete()
            return carla.VehicleVelocityControl(0, 0)

        # Check if the goal distance has been reached
        # get the current pose, and check if the distance traveled is greater than the target distance
        # TODO

        # Record sensor data every frame
        self.recorder.record_sensors(self.frame)

        # If images are present, record them and run ORBSLAM
        if self.frame % 2 == 0:
            # Capture images
            self.recorder.record_cameras(self.frame, input_data)

            # Run ORBSLAM
            # TODO
            orbslam_pose = {
                "x": 0,
                "y": 0,
                "z": 0,
                "yaw": 0,
                "pitch": 0,
                "roll": 0,
            }
            self.recorder.record_custom(self.frame, "orbslam", orbslam_pose)

        # Increment the frame counter
        self.frame += 1

        # Return the velocity control
        return carla.VehicleVelocityControl(self.linear_velocity, self.angular_velocity)

    def finalize(self):
        # Stop recording if still active
        self.recorder.stop()
