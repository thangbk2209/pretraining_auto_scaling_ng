import numpy as np
import time
import pickle
from tensorflow.keras.models import save_model
import threading
import copy
import multiprocessing
from config import *


class Particle:
    def __init__(self, types, mins, maxs, names, v_maxs):
        self.v_maxs = v_maxs  # np array: v_max of each particular dimensions
        self.types = types  # type of attribute: discrete, continuous
        self.mins = mins  # min value of domains
        self.maxs = maxs
        self.names = names  # name of attributes

        self.position = self.mins + (self.maxs - self.mins) * np.random.uniform(size=len(names))
        self.position = self._correct_position(self.position)
        self.velocity = v_maxs / 2 * np.random.uniform(-1, 1, size=len(names))

        self.pbest_position = self.position
        self.pbest_model = None
        self.pbest_value = float('inf')
        self.pbest_attribute = None
        self.pbest_test_error_inv = None

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

    def move(self):
        self.velocity = np.clip(self.velocity, -self.v_maxs, self.v_maxs)
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
    def __init__(self, fitness_fn, domain, n_particles,
                 max_w_old_velocity=0.9, min_w_old_velocity=0.4,
                 w_pbest=1.2, w_gbest=1.2):
        # domain: dictionary from config
        self.fitness_fn = fitness_fn
        self._parse_domain(domain)
        self.n_particles = n_particles
        self.v_maxs = (self.maxs - self.mins) / 2
        self.particles = self.create_particles()

        self.gbest_value = float('inf')
        self.gbest_model = None
        self.gbest_position = None
        self.gbest_test_error_inv = None
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
                Particle(self.types, self.mins, self.maxs, self.names, self.v_maxs)
            )
        return particles

    def evaluate_particle(self, particle):
        fitness, model, test_error_inv = self.fitness_fn(particle.decode_position())
        if fitness < particle.pbest_value:
            particle.pbest_position = particle.position
            particle.pbest_value = fitness
            particle.pbest_model = model
            particle.pbest_test_error_inv = test_error_inv
            particle.pbest_attribute = particle.decode_position()

    def _evaluate_particles(self, particles):
        for particle in particles:
            self.evaluate_particle(particle)

    def update_pbest_gbest(self, multithreading=True):
        # if multithreading:
        #     print('multithreading Mode')
        #
        #     # split particles for threads
        #     n_threads = int(multiprocessing.cpu_count() / 2)
        #     if n_threads > self.n_particles:
        #         n_threads = self.n_particles
        #     n_particles_per_thread = int(self.n_particles / n_threads)
        #
        #     threads = []
        #     for idx_thread in range(n_threads):
        #         start = idx_thread * n_particles_per_thread
        #         if idx_thread == n_threads - 1:
        #             end = n_threads
        #         else:
        #             end = start + n_particles_per_thread
        #         list_particles = self.particles[start:end]
        #         _thread = threading.Thread(target=self._evaluate_particles, args=(list_particles,))
        #         threads.append(_thread)
        #
        #     for thread in threads:
        #         thread.start()
        #
        #     for thread in threads:
        #         thread.join()
        if multithreading:
            print('multiprocessing.Pool Mode')
            multiprocessing.Pool(os.cpu_count()).map(self.evaluate_particle, self.particles)
        else:
            print('Single thread')
            for particle in self.particles:
                self.evaluate_particle(particle)

        for particle in self.particles:
            if particle.pbest_value < self.gbest_value:
                self.gbest_value = particle.pbest_value
                self.gbest_position = particle.pbest_position
                self.gbest_model = particle.pbest_model
                self.gbest_test_error_inv = particle.pbest_test_error_inv
                self.gbest_attribute = particle.pbest_attribute
                self.gbest_particle = particle

    def move_particles(self):
        for particle in self.particles:
            v1 = self.w_old_velocity * particle.velocity
            v2 = self.w_pbest * np.random.uniform() * (particle.pbest_position - particle.position)
            v3 = self.w_gbest * np.random.uniform() * (self.gbest_position - particle.position)
            particle.velocity = v1 + v2 + v3
            particle.move()

    def save_best_particle(self, iteration, losses):
        pso_results_dir = os.path.join(CORE_DATA_DIR, 'pso_results', RUN_ID)
        if not os.path.exists(pso_results_dir):
            os.mkdir(pso_results_dir)

        with open(os.path.join(pso_results_dir, 'config_result-losses_iter{}.pkl'.format(iteration)), 'wb') as out_file:
            pickle.dump(self.gbest_attribute, out_file)
            pickle.dump(losses, out_file)
        save_model(self.gbest_model, os.path.join(pso_results_dir, 'generator_iter{}.h5'.format(iteration)))

        # pickle error when save tf model in particle
        # with open(os.path.join(pso_results_dir, 'checkpoint_particles_iter{}'.format(iteration)), 'wb') as out_file:
        #     particles_copy = copy.deepcopy(self.particles)
        #     for particle in particles_copy:
        #         particle.pbest_model = None
        #     pickle.dump(particles_copy, out_file)

    def search(self, max_iter, step_save=2, early_stopping=10, multithreading=True):
        losses = []
        iteration = None
        for iteration in range(1, max_iter+1):
            self.w_old_velocity = self.max_w_old_velocity \
                                  - (iteration / max_iter) * (self.max_w_old_velocity - self.min_w_old_velocity)
            start_time = time.time()
            print('iteration: {}'.format(iteration))

            self.update_pbest_gbest(multithreading=multithreading)
            self.move_particles()
            losses.append(self.gbest_value)
            print('best fitness: {}, time: {}'.format(self.gbest_value, time.time() - start_time))

            if iteration % step_save == 0:
                self.save_best_particle(iteration, losses)

            if iteration > early_stopping:
                if losses[-1] == losses[-early_stopping]:
                    break

        self.save_best_particle(-1, losses)
        print('Best solution: iteration: {}, fitness: {}, test_error_inv: {}'
              .format(iteration, self.gbest_value, self.gbest_test_error_inv))
        return self.gbest_particle
