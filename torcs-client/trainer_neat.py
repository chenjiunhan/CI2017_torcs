"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import neat
import visualize
import numpy as np
import data_processor as dp

'''clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(5, 2), random_state=1)
X=[[-61, 25, 0.62, 0.64, 2, -35, 0.7, 0.65], [2,-5,0.58,0.7,-3,-15,0.65,0.52]]
y=[ [0.63, 0.64], [0.58,0.61] ]
clf.fit(X,y)'''
#exit(0)
torcs_inputs = dp.input_data.tolist()
#scaler = preprocessing.StandardScaler().fit(x)
#x = scaler.transform(x)
torcs_outputs = dp.output_data.tolist()

# 2-input XOR inputs and expected outputs.
#torcs_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
#torcs_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        show_one = True
        for xi, xo in zip(torcs_inputs, torcs_outputs):            
            output = net.activate(xi)
            if show_one:
                show_one = False
                #print(xi)
                #print(output)
            
            genome.fitness -= (output[0] - xo[0]) ** 2 + (output[1] - xo[1]) ** 2 + (output[2] - xo[2]) ** 2


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(torcs_inputs, torcs_outputs):        
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
    
