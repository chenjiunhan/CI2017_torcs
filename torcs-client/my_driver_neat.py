from pytocl.driver import Driver
from pytocl.car import State, Command
from pytocl.main import main

'''import _thread'''

import numpy as np
import os

'''import pyautogui'''

import neat
import visualize
import time

TRAIN = True
DEBUG = False
VERBOSE = True

# For simulate user input, not using now
'''def press_e(name, delay):
    print('sleep')    
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(delay)
    pyautogui.press('e')
    pyautogui.press('enter')
'''

class MyDriver(Driver):    
    def drive(self, carstate: State) -> Command:
                
        # count from race start
        global count_s

        # if race finished
        global termination

        command = Command()

        # * 3.6 to KM/H
        speed = carstate.speed_x * 3.6        
        x = [carstate.speed_x * 3.6, carstate.distance_from_center, carstate.angle] + list(carstate.distances_from_edge)
        if DEBUG:
            print('x', x)

        # normalize x
        '''x = tc.scaler.transform([x])'''

        y = self.output_net.activate(x)

        if DEBUG:
            print('y', y)
        
        command.accelerator = y[0]
        command.brake = y[1]
        command.steering = y[2]

        if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1
        
        # Termination condition
        # 1. if car speed < 10 after count_s > 100 
        # 2. hit wall
        # 3. off track

        count_s += 1
        if (count_s > 100 and abs(speed - 0.0) < 10) or \
           carstate.damage > 0 or \
           carstate.distance_from_center >= 1 or \
           carstate.distance_from_center <= -1:
            
            termination = True

        if termination:
            
            # fitness
            fitness_factor['raced_distance'] = carstate.distance_raced
            
            # Reset parameters
            termination = False
            count_s = 0         

            # Ask server to restart
            command.meta = 1
        
        return command
    
    def on_restart(self):

        print('-----------------Restart-------------------')

        # Terminate torcs client
        raise KeyboardInterrupt

        # Simulate user input
        '''_thread.start_new_thread(press_e, ('thread_e', 3))'''
        

def eval_genomes(genomes, config):
    global fitness_factor

    print('-------------------Evolution start-----------------------')
    for genome_id, genome in genomes:
        
        print('-------------------Start to race-----------------------')   
        
        # init MyDriver
        my_driver = MyDriver()

        # create feed forward network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # pass net to my_driver for computing output during race
        my_driver.output_net = net

        # start to race
        main(my_driver)

        # race fitness
        genome.fitness = fitness_factor['raced_distance']

        print('-------------------Race result-----------------------')
        print('Fitness:', genome.fitness)
        
        print('-------------------End race-----------------------') 

    print('-------------------Evolution end-----------------------')


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

    # Start to train
    winner = p.run(eval_genomes)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    #winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    #for xi, xo in zip(torcs_inputs, torcs_outputs):        
    #    output = winner_net.activate(xi)
    #    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


    # visualize training result
    '''node_names = {-1:'A', -2: 'B', 0:'A XOR B'}'''
    '''visualize.draw_net(config, winner, True, node_names=node_names)'''
    
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    
    # I don't know what's this
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')

    '''p.run(eval_genomes, 10)'''

if TRAIN:
    fitness_factor = {}
    termination = False
    count_s = 0
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)   
