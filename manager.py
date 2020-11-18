# from src.training import train
#
# train()

import time
from src.pso.pso import Space
from config import GanConfig
from src.pso.fitness_fn import fitness_function

gan_config = GanConfig()
pso_config = gan_config.PSO['pso_config']
domain = gan_config.PSO['domain']


if __name__ == '__main__':
    start_time = time.time()
    space = Space(fitness_function, domain, pso_config['n_particles'])
    space.search(
        max_iter=pso_config['max_iter'],
        step_save=pso_config['step_save'],
        early_stopping=pso_config['early_stopping'],
        multithreading=pso_config['multithreading']
    )
    training_time = time.time() - start_time
    print('Training in {:.2} hours'.format(training_time/(24*60*60)))
