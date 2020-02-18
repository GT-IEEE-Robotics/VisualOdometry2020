#!/usr/bin/env python3
"""
File:          pygame_viz.py
Author:        Binit Shah
Last Modified: Binit on 2/18
"""

import random
import pygame as pg
import particle_filter

def inch_to_pixels(val_in, offset=True):
    # 100in to 400 pixels
    if offset:
        return (val_in * 4) + 50
    else:
        return val_in * 4

if __name__ == "__main__":
    pg.init()
    grid = particle_filter.Grid()

    display = pg.display.set_mode((472, 280))
    display.fill((255, 255, 255))

    # Draw board
    pg.draw.rect(display, (0, 0, 0), (inch_to_pixels(0), inch_to_pixels(0),
                                      inch_to_pixels(grid.width_in, offset=False), inch_to_pixels(grid.height_in, offset=False)))

    # Draw walls
    for w in grid.walls:
        pg.draw.rect(display, (255, 255, 0), (inch_to_pixels(w[0]), inch_to_pixels(w[1]),
                                              inch_to_pixels(w[2] - w[0], offset=False), inch_to_pixels(w[3] - w[1], offset=False)))

    # Draw particles
    particles = grid.initialize_particles(500)
    for particle in particles:
        robot_pts = particle.vertices()
        robot_pts_pix = [(inch_to_pixels(robot_pts[0][0]), inch_to_pixels(robot_pts[0][1])),
                         (inch_to_pixels(robot_pts[1][0]), inch_to_pixels(robot_pts[1][1])),
                         (inch_to_pixels(robot_pts[2][0]), inch_to_pixels(robot_pts[2][1])),
                         (inch_to_pixels(robot_pts[3][0]), inch_to_pixels(robot_pts[3][1]))]
        pg.draw.polygon(display, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), robot_pts_pix)

    # Draw robot
    robot = particle_filter.Particle(9.5, 22.5, 0.0)
    robot_pts = robot.vertices()
    robot_pts_pix = [(inch_to_pixels(robot_pts[0][0]), inch_to_pixels(robot_pts[0][1])),
                     (inch_to_pixels(robot_pts[1][0]), inch_to_pixels(robot_pts[1][1])),
                     (inch_to_pixels(robot_pts[2][0]), inch_to_pixels(robot_pts[2][1])),
                     (inch_to_pixels(robot_pts[3][0]), inch_to_pixels(robot_pts[3][1]))]
    pg.draw.polygon(display, (181, 101, 29), robot_pts_pix)

    # Check collision
    print("initial_pos is_free: ", grid.is_free(robot.x_in, robot.y_in, robot.theta_deg))

    while True:
        pg.display.update()