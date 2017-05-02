import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib import animation

from math import pi as PI
from math import cos, sin, atan2, sqrt, exp

from scipy.stats.mstats import gmean

import numpy as np

from random import uniform, gauss

from itertools import chain


max_particles = 750
do_kld = False
sample_method = "kld"
if not do_kld:
    max_particles = 300


bin_size = 0.1 # twenty centimeters
angular_bin_size = 0.07 # ~12 degrees
num_samples_needed = 2
min_samples = 10
max_particles = max_particles

error = 0.3
z_score = 2.5 # 0.9993 percentile


class Particle(object):

    def __init__(self, x=0.0, y=0.0, theta=0.0):

        self.x = x
        self.y = y
        self.theta = theta#PI / 2.0

        self.weight = 1 # uniform weights at beginning!

    def re_init(self, x, y, theta):
        if do_kld:
            x = uniform(50, 60)
            y = uniform(50, 60)
            theta = uniform(theta-PI/4, theta+PI/4)
        self.x = x
        self.y = y
        self.theta = theta

        self.weight = 1

    def predict_from_odometry(self, delta_x, delta_y, delta_theta):

        # now lets get noise samples from zero-mean gaussian distributions. Define variance weights below
        alpha_1 = 0.2 # radial-radial

        alpha_2 = 0.1 # tang-tang
        alpha_3 = 0.05 # tang-radial
        alpha_4 = 0.05 # tang-angular

        alpha_5 = 1 # orientation noise

        radial_noise = gauss(0, alpha_1 * delta_x)# + gauss(0, alpha_2 * delta_phi)
        tangential_noise = (gauss(0, alpha_2 * abs(delta_y)) +
                            gauss(0, alpha_3 * abs(delta_x)) +
                            gauss(0, alpha_4 * abs(delta_theta)))

        theta_noise = gauss(0, alpha_5*abs(delta_theta))

        delta_x += radial_noise
        delta_y += tangential_noise
        delta_theta += theta_noise

        self.x += delta_x*cos(self.theta) - delta_y*sin(self.theta)
        self.y += delta_x*sin(self.theta) + delta_y*cos(self.theta)
        self.theta = (self.theta + delta_theta) % (2*PI)

    def predict_from_odometry_2(self, delta_x, delta_y, delta_theta):

        # now lets get noise samples from zero-mean gaussian distributions. Define variance weights below
        alpha_1 = 0.2 # radial-radial

        alpha_2 = 0.1 # tang-tang
        alpha_3 = 0.05 # tang-radial
        alpha_4 = 0.05 # tang-angular

        alpha_5 = 1 # orientation noise

        radial_noise = gauss(0, alpha_1 * delta_x)# + gauss(0, alpha_2 * delta_phi)
        tangential_noise = (gauss(0, alpha_2 * abs(delta_y)) +
                            gauss(0, alpha_3 * abs(delta_x)) +
                            gauss(0, alpha_4 * abs(delta_theta)))

        theta_noise = gauss(0, alpha_5*abs(delta_theta))

        delta_x += radial_noise
        delta_y += tangential_noise
        delta_theta += theta_noise

        new_x = self.x + delta_x*cos(self.theta) - delta_y*sin(self.theta)
        new_y = self.y + delta_x*sin(self.theta) + delta_y*cos(self.theta)
        new_theta = (self.theta + delta_theta) % (2*PI)

        return new_x, new_y, new_theta



    def find_weight(self, laser_unit_mat, laser_mags, map_lines):
        rotation_matrix = np.array([[cos(self.theta), -sin(self.theta)],
                                    [sin(self.theta), cos(self.theta)]])
        translation_matrix = np.array([[self.x, self.y]])


        laser_max_range = 4.01

        lasers_at_max_range = laser_max_range * laser_unit_mat.T

        # translate and rotate lasers to particle location
        max_lasers = np.matmul(rotation_matrix, lasers_at_max_range).T + translation_matrix
        # for each laser direction, find closest intersection point

        prob_of_each_laser = list()
        sub_sample = int(len(laser_mags) / 35)

        for i, (l_mag, (laser_x, laser_y)) in enumerate(zip(laser_mags, max_lasers)):
            if l_mag < 0.02 or (i % sub_sample) != 0:
                continue # the laser was out of range

            min_intersection_distance = None
            min_line = None
            for line in map_lines:
                dist = intersect_lines(self.x, self.y, laser_x, laser_y,
                                       line[0], line[1], line[2], line[3])

                if dist is None: # there is no intersection with this line!
                    continue

                if min_intersection_distance is None:
                    min_intersection_distance = dist
                    min_line = line

                if dist < min_intersection_distance:
                    min_intersection_distance = dist
                    min_line = line

            # now we know the distance of the closest intersection!
            if min_intersection_distance is None:
                # all walls are more than max-range away
                min_intersection_distance = 1.5

            min_intersection_distance *= laser_max_range
            laser_prob = laser_probability_function(l_mag, min_intersection_distance, 4.0)
            #gaussian_function(l_mag, min_intersection_distance) # min_int_dist is the mean!
            prob_of_each_laser.append(laser_prob)

        # use scipy geometric mean
        self.weight = gmean(prob_of_each_laser)

######### utility functions for observation likelihood ##################
def intersect_lines(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y):
    """
    line1 from p0 -> p1, and line2 from p2 -> p3
    p0 is the location of the robot
    """

    s10_x = p1x - p0x
    s10_y = p1y - p0y
    s32_x = p3x - p2x
    s32_y = p3y - p2y

    denom = (s10_x*s32_y) - (s32_x*s10_y)

    if denom == 0: return None
    denom_pos = denom > 0

    s02_x = p0x - p2x
    s02_y = p0y - p2y

    s = (s10_x*s02_y) - (s10_y*s02_x)

    if (s < 0) == denom_pos or (s > denom) == denom_pos: return None

    t = (s32_x*s02_y) - (s32_y*s02_x)

    if (t < 0) == denom_pos or (t > denom) == denom_pos: return None

    # thus there is an intersection. it happens t*|line1| away from p0,
    # so all we care about is t

    t = t / denom

    return t

def laser_probability_function(l_mag, closest_wall, laser_max_range):

    prob_of_hit = gaussian_function(l_mag, closest_wall)

    rand_weight = 0.005
    if l_mag < laser_max_range:
        prob_of_random = rand_weight * 1
    else:
        prob_of_random = 0.0

    max_range_weight = 0.25
    if l_mag < laser_max_range + 0.05 and l_mag > laser_max_range - 0.05:
        prob_of_max_range = max_range_weight * 1
    else:
        prob_of_max_range = 0.0

    short_hit_weight = 0.15
    if l_mag < closest_wall:
        prob_of_short_hit = short_hit_weight * exp(-1*l_mag)
    else:
        prob_of_short_hit = 0.0

    return prob_of_hit + prob_of_random + prob_of_max_range + prob_of_short_hit

def gaussian_function(x, mean, stddev=0.07):
    return exp( -(x - mean)**2 / (2 * (stddev)**2) )

######### utility functions for observation likelihood ##################

class Particles:
    '''
    This class is the filter itself. It maintains a list of samples (the particles defined in the above Particle class).
    For ease of use by the animation code, this class is an iterable to list through the particles.
    This returns the location of the robot with the location method. Currently this computes the mean of the coordinates
    of the list of particles.
    If the odometry tells us that we haven't moved, then we don't do the update from observation step.
    '''
    def __init__(self, num_particles, map_lines, x=0, y=0, theta=0):

        self.num_particles = num_particles
        self.particles = [Particle(x, y, theta) for i in range(num_particles)]
        self._iter_counter = 0

        self.movement_was_zero = False

        self.map_lines = map_lines

        self.num_lasers = None
        self.laser_min_theta = None
        self.laser_theta_delta = None
        self.laser_unit_mat = None

    # turn the Particles object into a list-like object!
    def __iter__(self):
        return self

    def next(self):
        if self._iter_counter == self.num_particles:
            self._iter_counter = 0 # reset the counting!
            raise StopIteration

        self._iter_counter += 1
        return self.particles[self._iter_counter - 1]

    def __getitem__(self, arg):
        return self.particles[arg]

    def __len__(self):
        return self.num_particles

    def re_initialize(self, x, y, theta):
        for p in self.particles:
            p.re_init(x, y, theta)

    def location(self):
        mean_x = mean_y = 0
        mean_theta = [] # slightly more complicated
        for p in self.particles:
            mean_x += p.x
            mean_y += p.y
            mean_theta.append((cos(p.theta), sin(p.theta)))

        mean_theta = ((sum(cos_t for cos_t, s in mean_theta)/self.num_particles,
                       sum(sin_t for c, sin_t in mean_theta)/self.num_particles))
        return mean_x / self.num_particles, mean_y / self.num_particles, atan2(mean_theta[1], mean_theta[0])

    def bounding_box(self):
        x, y, theta = self.location()
        max_x = max_y = x
        min_x = min_y = y
        for p in self.particles:
            if p.x > max_x:
                max_x = p.x
            if p.x < min_x:
                min_x = p.x
            if p.y > max_y:
                max_y = p.y
            if p.y < min_y:
                min_y = p.y

        laser_max_range = 4.05 # slightly more than the max range, just in case...

        return (max_x+laser_max_range, min_x-laser_max_range,
                max_y+laser_max_range, min_y-laser_max_range)

    def find_relevant_map_lines(self):
        max_x, min_x, max_y, min_y = self.bounding_box()

        relevant_lines = list()
        for line in self.map_lines:
            left_x = line[0]
            right_x = line[2]
            top_y = max(line[1], line[3])
            bot_y = min(line[1], line[3])
            if ((left_x < max_x and right_x > min_x) and
                (top_y > min_y and bot_y < max_y)):

                relevant_lines.append(line)

        #print len(relevant_lines)
        return relevant_lines


    def make_laser_updates(self, lasers):

        if self.num_lasers == None: # do this stuff only once!
            self.num_lasers = int(lasers[0])
            self.laser_min_theta = lasers[1]
            self.laser_theta_delta = lasers[3]

            self.laser_unit_mat = np.array([[cos(i*self.laser_theta_delta + self.laser_min_theta),
                                             sin(i*self.laser_theta_delta + self.laser_min_theta)]
                                            for i in range(self.num_lasers)])


        if self.movement_was_zero:
            return self.location()


        laser_magnitudes = np.array(lasers[4:])
        relevant_map_lines = self.find_relevant_map_lines()

        for p in self.particles:
            p.find_weight(self.laser_unit_mat, laser_magnitudes, relevant_map_lines)

        #print "sum of weight", sum(p.weight for p in self.particles)


        # after finding the weights, we need to resample
        #if not do_kld or do_kld:
        self.resample(sample_method)

        # after resampling, return the location
        return self.location()

    def resample(self, resample_method="low_var"):

        if resample_method == "low_var":
            return self.low_variance_resample()

        if resample_method == "kld":
            return self.kld_resample()


    def low_variance_resample(self):

        bin_bounds = list()
        sum_of_weights = 0
        for p in self.particles:
            sum_of_weights += p.weight
            bin_bounds.append(sum_of_weights)

        # systematic resampling
        resample_slice = sum_of_weights / self.num_particles
        random_number = uniform(0, resample_slice)

        resample_values = list()
        i = 0
        while random_number < sum_of_weights:
            current_particle_bound = bin_bounds[i]
            current_particle = self.particles[i]
            while random_number < current_particle_bound:
                resample_values.append((current_particle.x, current_particle.y, current_particle.theta))
                random_number += resample_slice
            i += 1

        # now change our list according to resample indicies!
        for (x, y, theta), particle in zip(resample_values, self.particles):
            particle.re_init(x, y, theta)

    def kld_resample(self):
        bins = set() # set of tuples, each tuple represents a pose bin (x, y, theta)

        num_samples = 0

        num_samples_needed = min_samples
        keep_sampling = lambda x: ((x < num_samples_needed) or (x < min_samples)) and (x < max_particles)

        particle_cum_weights = list()
        total_weights = 0.0
        for p in self.particles:
            total_weights += p.weight
            particle_cum_weights.append(total_weights)

        weight_inc = total_weights / self.num_particles
        if total_weights < 0.0001 or weight_inc < 0.0001:
            weight_inc = 1.0
            total_weights = 0.0
            for i, p in enumerate(self.particles):
                p.weight = weight_inc
                total_weights += weight_inc
                particle_cum_weights[i] = total_weights

        new_particles = list()

        random_number = uniform(0, weight_inc)
        i = 0
        while keep_sampling(num_samples):
            current_weight_lim = particle_cum_weights[i]
            if random_number < current_weight_lim:
                current_particle = self.particles[i]

                current_bin = (int(current_particle.x / bin_size),
                               int(current_particle.y / bin_size),
                               int(current_particle.theta / angular_bin_size))

                if current_bin not in bins:
                    bins.add(current_bin)

                    k = max(2, len(bins))

                    num_samples_needed = ((k-1.0) / (2.0*error)) * (1.0 - (2.0/(9.0*(k-1.0))) + sqrt(2.0/(9.0*(k-1.0)))*z_score)**3

                num_samples += 1
                new_particle = Particle(x=current_particle.x, y=current_particle.y, theta=current_particle.theta)
                new_particles.append(new_particle)

                random_number += weight_inc
            else:
                if random_number >= total_weights:
                    random_number = max(0.0, random_number - total_weights)
                i += 1
                if i >= self.num_particles:
                    i = 0

        #print len(bins)
        self.particles = new_particles
        self.num_particles = len(self.particles)



    def make_odometry_predictions(self, odometry):
        assert len(odometry) == 3

        if odometry[0] == odometry[1] == odometry[2] == 0:
            self.movement_was_zero = True
            return self.location()

        self.movement_was_zero = False

        for p in self.particles:
            p.predict_from_odometry(odometry[0], odometry[1], odometry[2])

        return self.location()




############# utilities for drawing the particles, laser dots, and path ########################################
def re_initialize_particles_and_images(particles, particle_images, truth_particle, center_x, center_y, angle):
    """re init the particles and their images"""

    print len(particles)
    assert len(particles) <= len(particle_images)

    particles.re_initialize(center_x, center_y, angle)

    # these are the same for all particles
    angle = (angle - PI/2) * (180 / PI)
    theta_1, theta_2 = angle, angle + 180

    # particles is a list of matplotlib Wedges
    for p_im in particle_images:
        p_im[0].set_center((center_x, center_y))
        p_im[0].set_theta1(theta_1)
        p_im[0].set_theta2(theta_2)

        p_im[1].set_center((center_x, center_y))
        p_im[1].set_theta1(theta_2)
        p_im[1].set_theta2(theta_1)

    truth_particle[0].set_center((center_x, center_y))
    truth_particle[0].set_theta1(theta_1)
    truth_particle[0].set_theta2(theta_2)
    truth_particle[1].set_center((center_x, center_y))
    truth_particle[1].set_theta1(theta_2)
    truth_particle[1].set_theta2(theta_1)

    return True


def find_laser_endpoints_for_display(truth_particle, location, laser_reading):

    loc_x, loc_y, loc_theta = location

    rot_mat = np.array([[cos(loc_theta), -sin(loc_theta)],
                        [sin(loc_theta), cos(loc_theta)]])

    trans_mat = np.array([loc_x, loc_y])

    num_rays_max = int(laser_reading[0])
    min_theta = laser_reading[1]
    max_theta = laser_reading[2]

    theta_diff = laser_reading[3]
    laser_unit_mat = list()
    for i in range(num_rays_max):
        i_laser_theta = i * theta_diff + min_theta
        laser_unit_mat.append((cos(i_laser_theta), sin(i_laser_theta)))

    laser_unit_mat = np.array(laser_unit_mat)
    laser_magnitude = np.array([laser_reading[4:]])

    laser_locs = laser_magnitude * laser_unit_mat.T # r_i * <cos(theta_i), sin(theta_i)> = <x_i, y_i> in robot coords

    real_laser_locs = np.matmul(rot_mat, laser_locs).T + trans_mat # rotate and translate into map coords

    # only return non-null values, since this is for display
    laser_xs, laser_ys = zip(*[(lxs, lys) for lxs, lys in
                               [lasers for i, lasers in enumerate(real_laser_locs) if laser_reading[4+i] > 0.02]])



    return laser_xs, laser_ys

def update_particle_images(particles, particle_images):
    """update the images based on the mathy particles"""

    assert len(particles) <= len(particle_images)

    for i in range(len(particle_images)):
        if i < len(particles):
            p = particles[i]
            p_im = particle_images[i]

            angle = (p.theta - PI/2) * (180 / PI)
            theta_1, theta_2 = angle, angle + 180

            p_im[0].set_visible(True)
            p_im[0].set_center((p.x, p.y))
            p_im[0].set_theta1(theta_1)
            p_im[0].set_theta2(theta_2)

            p_im[1].set_visible(True)
            p_im[1].set_center((p.x, p.y))
            p_im[1].set_theta1(theta_2)
            p_im[1].set_theta2(theta_1)

        else:
            particle_images[i][0].set_visible(False)
            particle_images[i][1].set_visible(False)

    return True



def create_particle_images(particles, radius=0.4):

    particle_images = list()
    for p in particles:
        x = p.x
        y = p.y

        angle = (p.theta - PI/2) * (180 / PI)
        theta1, theta2 = angle, angle + 180
        w1 = Wedge((x, y), radius, theta1, theta2, fc='w')
        w2 = Wedge((x, y), radius, theta2, theta1, fc='k')
        p_im = (w1, w2)

        particle_images.append(p_im)


    x, y, theta = particles.location()
    angle = (theta - PI/2) * (180 / PI)
    theta1, theta2 = angle, angle + 180
    truth_particle_1 = Wedge((x, y), radius, theta1, theta2, fc='w')
    truth_particle_2 = Wedge((x, y), radius, theta2, theta1, fc='r')
    truth_particle = (truth_particle_1, truth_particle_2)

    return particle_images, truth_particle

############# utilities for drawing the particles, laser dots, and path ########################################

############ code for reading in data files ########################
def read_map_file(filename="data/map.txt"):
    with open(filename) as map_file:
        map_lines_list = []
        counter = 0
        for line_segment in map_file:
            line_1_x, line_1_y, line_2_x, line_2_y = (float(coord) for coord in line_segment.split(','))

            # guarantees that first point is the left point
            if line_1_x < line_2_x:
                map_lines_list.append((line_1_x, line_1_y, line_2_x, line_2_y))
            else:
                map_lines_list.append((line_2_x, line_2_y, line_1_x, line_1_y))

    return map_lines_list


def read_data_file(filename="assignment-2-data/robot-data.log", restart=False):

    with open(filename) as data_log:

        for data_line in data_log:
            #print "doing a thing"
            data_line = data_line.strip().split(" ")
            return_data = [data_line[0]]
            if data_line[0] == "I":
                return_data.extend([float(d) for d in data_line[2:]])

            else:
                return_data.extend([float(d) for d in data_line[1:]])

            yield return_data

############ code for reading in data files ########################

#### code to print speed of particle filter #############
def format_interval(t):
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    if h:
        return '%d:%02d:%02d' % (h, m, s)
    else:
        return '%02d:%02d' % (m, s)


def format_meter(n, total, elapsed):
    # n - number of finished iterations
    # total - total number of iterations, or None
    # elapsed - number of seconds passed since start
    if n > total:
        total = None

    elapsed_str = format_interval(elapsed)
    rate = '%5.2f' % (n / elapsed) if elapsed else '?'

    if total:
        frac = float(n) / total

        N_BARS = 10
        bar_length = int(frac*N_BARS)
        bar = '#'*bar_length + '-'*(N_BARS-bar_length)

        percentage = '%3d%%' % (frac * 100)

        left_str = format_interval(elapsed / n * (total-n)) if n else '?'

        return '|%s| %d/%d %s [elapsed: %s left: %s, %s iters/sec]' % (
            bar, n, total, percentage, elapsed_str, left_str, rate)

    else:
        return '%d [elapsed: %s, %s iters/sec]' % (n, elapsed_str, rate)


class StatusPrinter(object):
    def __init__(self, file):
        self.file = file
        self.last_printed_len = 0

    def print_status(self, s):
        self.file.write('\r'+s+' '*max(self.last_printed_len-len(s), 0))
        self.file.flush()
        self.last_printed_len = len(s)

########## code to print speed of particle filter ##################



if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # draw the map!
    map_lines_list = read_map_file()
    for x1, y1, x2, y2 in map_lines_list:
        ax.plot([x1, x2], [y1, y2], linestyle='solid', color='blue')

    x, y, theta = 40, 40, 0
    particles = Particles(num_particles=max_particles, map_lines=map_lines_list, x=x, y=y, theta=theta)
    particle_images, truth_particle = create_particle_images(particles)

    for p_im in particle_images:
        ax.add_patch(p_im[0])
        ax.add_patch(p_im[1])

    ax.add_patch(truth_particle[0])
    ax.add_patch(truth_particle[1])


    path, = ax.plot([],[], linestyle='solid', color='g')
    lasers, = ax.plot([], [], markersize=3, linestyle='None', marker='.', markeredgecolor='m')

    import time
    loc_str = lambda x,y,theta: "center = {x:.3f},{y:.3f}, orientation = {theta:.3f}".format(x=x, y=y, theta=theta)
    loc_text = ax.text(0, 15, loc_str(x, y, theta))

    time_str = lambda x: "timestamp = {time:.2f}".format(time=x)
    time_text = ax.text(0, 12, time_str(0.))

    num_particle_str = lambda x: "{num_p}".format(num_p=x)
    num_part_text = ax.text(35, 12, num_particle_str(0))

    data_log = read_data_file()


    start_time = time.time()
    last_time = start_time
    def init_anim():

        global start_time, last_time
        start_time = time.time()
        last_time = start_time

        re_initialize_particles_and_images(particles, particle_images, truth_particle, 40, 40, 0)
        loc_text.set_text(loc_str(40, 40, 0))
        time_text.set_text("Restarting!!!!!!")

        path.set_data([],[])
        lasers.set_data([], [])

        global data_log
        data_log = read_data_file(restart=True)

        artists_to_return = list(chain(*particle_images+[truth_particle]))
        artists_to_return.extend([time_text, loc_text, num_part_text, path, lasers])
        return artists_to_return

    import sys
    sp = StatusPrinter(sys.stderr)
    sp.print_status(format_meter(0, 30073, 0))

    def animate(i):
        global start_time, last_time
        cur_time = time.time()
        if cur_time - last_time > 0.5:
            sp.print_status(format_meter(i/3, 10025, cur_time - start_time))
            last_time = cur_time

        next_data = data_log.next()
        update_type = next_data[0]
        if update_type != "I":
            time_text.set_text(time_str(next_data[1]))

        if update_type == "O":
            x, y, theta = particles.make_odometry_predictions(next_data[2:]) # pass in only x, y, theta

            angle = (theta - PI/2) * (180 / PI)
            theta1, theta2 = angle, angle + 180
            truth_particle[0].set_center((x, y))
            truth_particle[0].set_theta1(theta1)
            truth_particle[0].set_theta2(theta2)
            truth_particle[1].set_center((x, y))
            truth_particle[1].set_theta1(theta2)
            truth_particle[1].set_theta2(theta1)

            x_data, y_data = path.get_data()
            x_data = list(x_data)
            y_data = list(y_data)
            x_data.append(x)
            y_data.append(y)
            path.set_data(x_data, y_data)

            loc_text.set_text(loc_str(x, y, theta))
            update_particle_images(particles, particle_images)

        if update_type == "L":
            particles.make_laser_updates(next_data[2:])
            update_particle_images(particles, particle_images)
            laser_xs, laser_ys = find_laser_endpoints_for_display(truth_particle, particles.location(), next_data[2:])
            lasers.set_data(laser_xs, laser_ys)
            num_part_text.set_text(num_particle_str(len(particles)))

        elif update_type == "I": # this is the initialization step
            pass
            re_initialize_particles_and_images(particles, particle_images, truth_particle,
                                               next_data[1], next_data[2], next_data[3])

            loc_text.set_text(loc_str(next_data[1], next_data[2], next_data[3]))

        artists_to_return = list(chain(*particle_images+[truth_particle]))
        artists_to_return.extend([time_text, loc_text, num_part_text, path, lasers])
        return artists_to_return

    anim = animation.FuncAnimation(fig, animate, init_func=init_anim, frames=30073,
                                   interval=1, repeat_delay=20000, blit=True)

    import datetime
    filename = "particles_run_at_" + str(datetime.datetime.now()) + ".mp4"
    #anim.save(filename, fps=30, dpi=200, extra_args=['-vcodec', 'libx264'])

    plt.show()
