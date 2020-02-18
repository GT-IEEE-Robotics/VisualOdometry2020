import numpy as np
import random
import math

class Particle:
    def __init__(self, x, y, theta_deg):
        self.x_in = x
        self.y_in = y
        self.theta_rad = theta_deg * (np.pi / 180.0)
        self.robot_width_in = 9.0
        self.robot_height_in = 10.5

    def vertices(self):
        robot_pts = [[self.x_in + (0.5 * self.robot_width_in * math.cos(self.theta_rad) - 0.5 * self.robot_height_in * math.sin(self.theta_rad)),
                      self.y_in + (0.5 * self.robot_width_in * math.sin(self.theta_rad) + 0.5 * self.robot_height_in * math.cos(self.theta_rad))],
                     [self.x_in + (-0.5 * self.robot_width_in * math.cos(self.theta_rad) - 0.5 * self.robot_height_in * math.sin(self.theta_rad)),
                      self.y_in + (-0.5 * self.robot_width_in * math.sin(self.theta_rad) + 0.5 * self.robot_height_in * math.cos(self.theta_rad))],
                     [self.x_in + (-0.5 * self.robot_width_in * math.cos(self.theta_rad) + 0.5 * self.robot_height_in * math.sin(self.theta_rad)),
                      self.y_in + (-0.5 * self.robot_width_in * math.sin(self.theta_rad) - 0.5 * self.robot_height_in * math.cos(self.theta_rad))],
                     [self.x_in + (0.5 * self.robot_width_in * math.cos(self.theta_rad) + 0.5 * self.robot_height_in * math.sin(self.theta_rad)),
                      self.y_in + (0.5 * self.robot_width_in * math.sin(self.theta_rad) - 0.5 * self.robot_height_in * math.cos(self.theta_rad))]]
        return robot_pts

class Grid:
    """Defines the Southeastcon grid. Origin at bottomleft."""
    def __init__(self):
        self.width_in = 93.0
        self.height_in = 45.0
        # (bottomleft_corner x, y, topright_corner x, y)
        self.walls = [(19.255, 0, 20.005, 12), (30.005, 0, 30.755, 12), (40.775, 0, 41.505, 12), (51.505, 0, 52.255, 12), (62.255, 0, 63.005, 12), (73.005, 0, 73.755, 12),
                      (19.255, 33.0, 20.005, 45.0), (30.005, 33.0, 30.755, 45.0), (40.775, 33.0, 41.505, 45.0), (51.505, 33.0, 52.255, 45.0), (62.255, 33.0, 63.005, 45.0), (73.005, 33.0, 73.755, 45.0),
                      (-1.5, -1.5, 94.5, 0), (-1.5, 0, 0, 45.0), (93.0, 0, 94.5, 45.0), (-1.5, 45.0, 94.5, 46.5)]

    def is_overlapping(self, robot, wall):
        """
        * Helper function to determine whether there is an intersection between the two polygons described
        * by the lists of vertices. Uses the Separating Axis Theorem
        * https://stackoverflow.com/a/56962827
        *
        * @return true if there is any intersection between the 2 polygons, false otherwise
        """
        a = robot.vertices()
        b = [[wall[0], wall[1]], [wall[0], wall[3]], [wall[2], wall[3]], [wall[2], wall[1]]]
        polygons = [a, b];
        minA, maxA, projected, i, i1, j, minB, maxB = None, None, None, None, None, None, None, None

        for i in range(len(polygons)):

            # for each polygon, look at each edge of the polygon, and determine if it separates
            # the two shapes
            polygon = polygons[i];
            for i1 in range(len(polygon)):

                # grab 2 vertices to create an edge
                i2 = (i1 + 1) % len(polygon);
                p1 = polygon[i1];
                p2 = polygon[i2];

                # find the line perpendicular to this edge
                normal = { 'x': p2[1] - p1[1], 'y': p1[0] - p2[0] };

                minA, maxA = None, None
                # for each vertex in the first shape, project it onto the line perpendicular to the edge
                # and keep track of the min and max of these values
                for j in range(len(a)):
                    projected = normal['x'] * a[j][0] + normal['y'] * a[j][1];
                    if (minA is None) or (projected < minA): 
                        minA = projected

                    if (maxA is None) or (projected > maxA):
                        maxA = projected

                # for each vertex in the second shape, project it onto the line perpendicular to the edge
                # and keep track of the min and max of these values
                minB, maxB = None, None
                for j in range(len(b)): 
                    projected = normal['x'] * b[j][0] + normal['y'] * b[j][1]
                    if (minB is None) or (projected < minB):
                        minB = projected

                    if (maxB is None) or (projected > maxB):
                        maxB = projected

                # if there is no overlap between the projects, the edge we are looking at separates the two
                # polygons, and we know there is no overlap
                if (maxA < minB) or (maxB < minA):
                    return False;

        return True

    def is_free(self, x_in, y_in, theta_deg):
        free = True
        for w in self.walls:
            if self.is_overlapping(Particle(x_in, y_in, theta_deg), w):
                free = False
        return free

    def random_free_pose(self):
        while True:
            x = random.uniform(0, self.width_in)
            y = random.uniform(0, self.height_in)
            theta = random.uniform(0, 360)
            if self.is_free(x, y, theta):
                return (x, y, theta)

    def initialize_particles(self, num_particles):
        return [Particle(*self.random_free_pose()) for _ in range(num_particles)]

class ParticleFilter:
    def __init__(self, particles, robot, grid, initial_time):
        self.particles = particles
        self.robot = robot
        self.grid = grid
        self.prev_time = initial_time
        self.R = .04299 # radius of wheel in meters
        self.L = 0.1    # distance to single integrator point in meters
        self.D = 0.22   # wheel base in meters
        self.translation_sigma = 0.02
        self.heading_sigma = 2

    def motion_update(self, odom, current_time):
        """
        Applies a motion update to all particles based on wheel odometry.

        @param odom wheel odometry tuple in form (rmotor_vel, lmotor_vel)
        """
        omega_right, omega_left = odom

        for particle in self.particles:
            xdot = omega_right * (self.R/2 * math.cos(particle.theta_rad) - self.D/self.L * math.sin(particle.theta_rad)) + \
                   omega_left * (self.R/2 * math.cos(particle.theta_rad) + self.D/self.L * math.sin(particle.theta_rad)) + \
                   random.gauss(0.0, self.translation_sigma)
            ydot = omega_right * (self.R/2 * math.sin(particle.theta_rad) + self.D/self.L * math.cos(particle.theta_rad)) + \
                   omega_left * (self.R/2 * math.sin(particle.theta_rad) - self.D/self.L * math.cos(particle.theta_rad)) + \
                   random.gauss(0.0, self.translation_sigma)
            thetadot = omega_right * self.R/self.D - omega_left * self.R/self.D + random.gauss(0.0, self.heading_sigma)

            particle.x_in += xdot * (current_time - self.prev_time)
            particle.y_in += ydot * (current_time - self.prev_time)
            particle.theta_rad += thetadot * (current_time - self.prev_time)
            particle.theta_rad = particle.theta_rad % (2 * np.pi)

    def measurement_update(self, markers_list):
        pass

    def compute_mean_pose(self):
        """
        @return (x, y, theta, confidence)
        """
        pass
