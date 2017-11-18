from pytocl.driver import Driver
from pytocl.car import State, Command
from pytocl.main import main

import trainer_chen as tc
import trainer2_chen as tc2
import numpy as np

fitness_factor = {}
termination = False
count_t = 0
class MyDriver(Driver):    
    def drive(self, carstate: State) -> Command:
        command = Command()
        speed = carstate.speed_x * 3.6

        x = [carstate.speed_x * 3.6, carstate.distance_from_center, carstate.angle] + list(carstate.distances_from_edge)
        #x = tc.scaler.transform([x])
        print(x)
        x = [x]        
        y = tc.clf.predict(x)
        y = y[0]
        print('y1', y)
        #x = [carstate.speed_x * 3.6, carstate.distance_from_center, carstate.angle] + list(carstate.distances_from_edge)
        #x2 = np.array([x]).reshape(1, len(x))
        #y2 = tc2.esn.predict(x2)
        #y2 = y2[0]
        #print('y2', y2)
        
        command.accelerator = y[0]
        #command.accelerator = 1
        command.brake = y[1]
        #command.brake = 0
        if speed > 100:
            command.steering = y[2]
        else:
            command.steering = y[2]
            #command.steering = (y[2] + y2[2])/2

        if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1
        #command.gear = 1
        #if carstate.speed_x > 5:
        #
        count += 1
        if count > 10000:
            termination = True
        if termination:
            fitness_factor['raced_distance'] = carstate.raced_distance
            command.meta = 1
            termination = False
            count = 0            
        #self.steer(carstate, 0.0, command)
        #if command.brake < 0.5 or carstate.speed_x * 3.6 < 30:
        #    command.brake = 0
        #if carstate.speed_x < 50:
        #    command.brake = 0
        return command
    def on_restart(self):        
        main(MyDriver())

torcs_inputs = dp.input_data.tolist()
#scaler = preprocessing.StandardScaler().fit(x)
#x = scaler.transform(x)
#torcs_outputs = dp.output_data.tolist()
# 2-input XOR inputs and expected outputs.
#torcs_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
#torcs_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

def eval_genomes(genomes, config):
    global fitness_factor

    for genome_id, genome in genomes:
        genome.fitness = 1
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        show_one = True
        '''#for xi, xo in zip(torcs_inputs, torcs_outputs):            
        for xi in zip(torcs_inputs):
            output = net.activate(xi)
            if show_one:
                show_one = False
                #print(xi)
                #print(output)'''
            
        genome.fitness -= fitness_factor['raced_distance']


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

    #node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    #visualize.draw_net(config, winner, True, node_names=node_names)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
