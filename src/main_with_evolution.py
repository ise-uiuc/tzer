import argparse
from tzer import evolution
from tzer.evolution.evolution import Evolution
from tzer.evolution.fitness import FitnessElites, MAX
from tzer.relay_seeds import MODEL_SEEDS
from tzer.report import Reporter

target_seeds = MODEL_SEEDS[4::]

seed = target_seeds[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool-size', type=int, default=50, help='set pool size')
    parser.add_argument('--max-time', type=int, default=60*60, help='set max time of fuzzing')
    parser.add_argument('--max-generations', type=int, default=1000, help='set generations of fuzzing')
    parser.add_argument('--folder', type=str, help='bug report folder')

    args = parser.parse_args()

    reporter = Reporter(args.folder)
    evolution = Evolution(reporter)
    evolution.set_population_size(args.pool_size)
    evolution.set_max_generations(args.max_generations)
    evolution.create_genotypes(seed)
    evolution.set_fitness_type(MAX, 10000000.0)
    evolution.set_fitness_selections(FitnessElites(evolution.fitness_list, 0.5))
    evolution.run(max_time=args.max_time)
