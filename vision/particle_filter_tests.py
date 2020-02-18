#!/usr/bin/env python3
"""
File:          particle_filter_tests.py
Author:        Binit Shah
Last Modified: Binit on 2/18
"""

import sim
import cv2
import particle_filter

CONFIDENCE_THRES = 0.8

if __name__ == "__main__":
    c = sim.SimConfig("")
    sim.start(c)

    grid = particle_filter.Grid()
    particles = grid.initialize_particles(5000)
    robot = particle_filter.Particle(9.5, 22.5, 0.0)
    pfilter = particle_filter.ParticleFilter(particles, robot, grid, sim.get_time())

    while True:
        pfilter.motion_update(sim.read_robot_vels(), sim.get_time())
        # for p in pfilter.particles:
        #     print(p.x_in, p.y_in, p.theta_rad)
        #     break

        # raw_img = cv2.cvtColor(sim.read_robot_cam(), cv2.COLOR_BGR2RGB)
        # cv2.imshow("raw", raw_img)
        # cv2.waitKey(1)
        # landmarks = find_landmarks(raw_img)
        # pfilter.measurement_update(landmarks)

        # x, y, theta, confidence = pfilter.compute_mean_pose()
        # if confidence > CONFIDENCE_THRES:
        #     print(f"converged pose: ({x}, {y}, {theta})")
