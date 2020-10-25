import numpy as np
from time import time


class Particle:
    def __init__(self, types, mins, maxs, names, v_max):
        self.v_max = v_max
        self.types = types  # type of attribute: discrete, continuous
        self.mins = mins  # min value of domains
        self.maxs = maxs
        self.names = names  # name of attributes

        self.position = self.mins + (self.maxs - self.mins) * np.random.uniform(size=len(names))
        self.position = self._correct_position(self.position)
        self.velocity = np.random.uniform(-1, 1, len(names))

        self.pbest_position = self.position
        self.pbest_model = None
        self.pbest_value = float('inf')
        self.pbest_attribute = None

    def _correct_position(self, position):
        for i, _type in enumerate(self.types):
            if _type == 'discrete':
                position[i] = round(position[i])
        return position

    def decode_position(self, position=None):
        if position is None:
            position = self.position
        result = {}
        for i, _type in enumerate(self.types):
            if _type == 'discrete':
                result[self.names[i]] = int(round(position[i]))
            else:
                result[self.names[i]] = position[i]
        return result

    def decode_position_2(self, position):
        result = {}
        for i, _type in enumerate(self.types):
            if _type == 'discrete':
                result[self.names[i]] = int(position[i])
            else:
                result[self.names[i]] = position[i]
        return result

    def move(self):
        self.velocity = np.clip(self.velocity, -self.v_max, self.v_max)
        self.position = self.position + self.velocity
        self.position = self._correct_position(self.position)
        self.position = np.clip(self.position, self.mins, self.maxs)

    # def evaluate(self, fitness_fn):
    #     fitness, model = fitness_fn(self.decode_position(self.position))
    #     if fitness < self.pbest_value:
    #         self.pbest_position = self.position
    #         self.pbest_value = fitness
    #         self.pbest_model = model


class Space:
    def __init__(self, fitness_fn, domain, n_particles, v_max=float('inf'),
                 max_w_old_velocity=0.9, min_w_old_velocity=0.4,
                 w_pbest=1.2, w_gbest=1.2):
        # domain: dictionary from config
        self.fitness_fn = fitness_fn
        self._parse_domain(domain)
        self.n_particles = n_particles
        self.v_max = v_max
        self.particles = self.create_particles()

        self.gbest_value = float('inf')
        self.gbest_model = None
        self.gbest_position = None
        self.gbest_attribute = None
        self.gbest_particle = None

        self.max_w_old_velocity = max_w_old_velocity
        self.min_w_old_velocity = min_w_old_velocity
        self.w_old_velocity = None
        self.w_pbest = w_pbest
        self.w_gbest = w_gbest

    def _parse_domain(self, domain):
        names = []
        types = []
        mins = []
        maxs = []

        for attr in domain:
            names.append(attr['name'])
            types.append(attr['type'])
            mins.append(attr['domain'][0])
            maxs.append(attr['domain'][1])

        self.names = names
        self.types = types
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)

    def create_particles(self):
        particles = []
        for i in range(self.n_particles):
            particles.append(
                Particle(self.types, self.mins, self.maxs, self.names, self.v_max)
            )
        return particles

    def evaluate_particle(self, particle):
        fitness, model = self.fitness_fn(particle.decode_position())
        if fitness < particle.pbest_value:
            particle.pbest_position = particle.position
            particle.pbest_value = particle.pbest_value
            particle.pbest_model = model
            particle.pbest_attribute = particle.decode_position()

    def update_pbest_gbest(self):
        for particle in self.particles:
            self.evaluate_particle(particle)

        for particle in self.particles:
            if particle.pbest_value < self.gbest_value:
                self.gbest_value = particle.pbest_value
                self.gbest_position = particle.pbest_position
                self.gbest_model = particle.pbest_model
                self.gbest_attribute = particle.pbest_attribute
                self.gbest_particle = particle

    def move_particles(self):
        for particle in self.particles:
            v1 = self.w_old_velocity * particle.velocity
            v2 = self.w_pbest * np.random.uniform() * (particle.pbest_position - particle.position)
            v3 = self.w_gbest * np.random.uniform() * (self.gbest_position - particle.position)
            particle.velocity = v1 + v2 + v3
            particle.move()

    # TODO: implement save_best_particle function
    def save_best_particle(self):
        return None

    def search(self, max_iter, step_save=2):
        losses = []
        iteration = None
        for iteration in range(1, max_iter+1):
            self.w_old_velocity = self.max_w_old_velocity \
                                  - (iteration / max_iter) * (self.max_w_old_velocity - self.min_w_old_velocity)
            start_time = time()
            print('iteration: {}'.format(iteration))

            self.update_pbest_gbest()
            self.move_particles()
            losses.append(self.gbest_value)
            print('best fitness: {}, time: {}'.format(self.gbest_value, time() - start_time))
            if iteration % step_save == 0:
                self.save_best_particle()
        self.save_best_particle()
        print('Best solution: iteration: {}, fitness: {}'.format(iteration, self.gbest_value))
        return self.gbest_particle
