# TODO
- [ ] Remove anything that is not directly required
    - [X] Remove FastSAM
    - [X] AprilTag
    - [X] Corresponding Tests
- [ ] Implement "simple" orbslam version
    - Decide how to handle data collection when orbslam fails? (Maybe just collect data and see if this even happens to begin with)
- [ ] Need to include map preset information in the recording file
    - This may require updating the Recorder
- [ ] Look at how the integrated playback tool works
- [ ] Create straight line agent
- [ ] Create circle agent
- [ ] Figure out how to change map presets
- [ ] Figure out how to change sun position
- [ ] Figure out how to change starting position
- [ ] Create batching script to run and record data sets automatically

# Captured Data
- [ ] Front and rear stereo cameras (every other frame)
- [ ] Ground truth pose data (every frame?)
- [ ] ORBSLAM pose estimates
- [ ] IMU Data?

# Run Parameters
- [ ] Preset 1, 20m straight line (over what time span?)
- [ ] Preset 2, 20m straight line (over what time span?)
- [ ] Preset 3, 20m straight line (over what time span?)
- [ ] Preset 4, 20m straight line (over what time span?)
