In order to run my particle filter, use the command
>$ python particle_filter.py

There are only dependencies on matplotlib, numpy, and scipy.

This will run if the map and data logs are in the assignment-2-data directory which is in the same directory as the file particle_filter.py. These paths can be changed in the methods read_map_file() and read_data_file() respectively.

If you would like to save a run of the code as a .mp4 (not recommended) simply uncomment the line anim.save(…) at the end of particle_filter.py.

If you would like to change the number of particles in the filter, this value is set in the constructor of Particles (search for “num_particles=“)

If you would like to change the number of lasers readings sub sampled, search for sub_sample. The number of lasers is the denominator of sub_sample = int(len(laser_mags) / 35) so 35 is the number of readings examined in this case.

If you would like to change parameter of the motion model, search for alpha_1 to put you in the right section of the code.

If you would like to change parameters of the laser model, search for the function laser_probability_function()

The particle filter is implemented with the Particles class. The next frame is fetched in the animation function, which determines the type of frame and passes the information to a Particles instance, and uses the output to draw the new state of the world.

Each particle is implemented with the Particle class. The Particles class maintains a list of instances of the Particle class, and does prediction, update, and resampling on this list for each frame as is appropriate.

In the animation, each particle is represented by a circle with a black half and white half. The dividing line of the circle is perpendicular to the heading of the particle, ie the white half points forward.
There is also one particle with a red back-half. This particle is what the particle filter returns as the location of the robot (currently the mean of the particles). The green line traces out the path of the robot, so it follows the red particle. The magenta dots are where the laser beams end up in the frame of the red particle.