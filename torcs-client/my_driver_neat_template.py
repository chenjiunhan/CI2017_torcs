from pytocl.driver import Driver
from pytocl.car import State, Command
from pytocl.main import main
import pytocl.main

'''import _thread'''

import numpy as np
import os


import neat
import visualize
import time

TRAIN = True
RETRAIN = True
if TRAIN == False and RETRAIN == True:
    print('TRAIN is False, but RETRAIN is True!')
    exit(0)

RETRAIN_FROM_GENERATION = 6

DEBUG = False
VERBOSE = True
ACC = True
ACC_value = 2

# global variable
fitness_factor = {}
termination = False
count_s = 0

if ACC:
    import pyautogui
    def press_acc():
        for i in range(ACC_value):
            pyautogui.press('+')
# For simulate user input, not using now
'''def press_e(name, delay):
    print('sleep')    
    #pyautogui.hotkey('ctrl', 'c')

    pyautogui.press('esc')
    time.sleep(delay)
    pyautogui.press('enter')'''


class MyDriver(Driver):    
    def drive(self, carstate: State) -> Command:
                
        # count from race start
        global count_s
        global ACC        
        global client
        global TRAIN
        # if race finished
        global termination

        count_s += 1
        
        command = Command()

        # * 3.6 to KM/H
        speed = carstate.speed_x * 3.6

        # calculate top speed
        if not hasattr(self, 'top_speed'):
            self.top_speed = 0.0

        if speed > self.top_speed:
            self.top_speed = speed
        
        # calculate average speed
        if not hasattr(self, 'average_speed'):
            self.average_speed = 0.0
 
        self.average_speed = self.average_speed * (count_s - 1) / float(count_s) + speed / float(count_s)

        # prepare x
        x = [carstate.speed_x * 3.6, carstate.distance_from_center, carstate.angle] + list(carstate.distances_from_edge)
        
        if DEBUG:
            print('x', x)

        # normalize x
        '''x = tc.scaler.transform([x])'''

        # create output
        y = self.output_net.activate(x)

        if DEBUG:
            print('y', y)
        
        # assign output
        command.accelerator = y[0]
        command.brake = y[1]
        command.steering = y[2]

        # automatically switch gear
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
        print(carstate)
        if TRAIN:
            if (carstate.current_lap_time > 10 and abs(speed - 0.0) < 2) or \
               carstate.damage > 0 or \
               carstate.distance_from_center >= 1 or \
               carstate.distance_from_center <= -1:
                print('Termination!!!!!!!!!!!!!!')
                termination = True
                
            if ACC and not hasattr(self, 'ACC_flag'):
                self.ACC_flag = True
                press_acc()               

            if termination:
                
                # fitness
                fitness_factor['raced_distance'] = carstate.distance_raced
                fitness_factor['top_speed'] = self.top_speed
                fitness_factor['average_speed'] = self.average_speed
                print('fitness_factor:', fitness_factor)
                # Reset parameters
                termination = False
                count_s = 0         

                # Ask server to restart
                command.meta = 1
        
        return command
    
    def on_restart(self):

        print('-----------------Restart-------------------')
        
        raise KeyboardInterrupt
        
        # Simulate user input
        '''_thread.start_new_thread(press_e, ('thread_e', 3))'''
        
generation = 0
def eval_genomes(genomes, config):
    global fitness_factor
    global ACC
    global generation
    print('parent generation:', generation)
    print('-------------------Evolution start-----------------------')
    for genome_id, genome in genomes:
        
        #time.sleep(2)
        print('-------------------Start to race-----------------------')
        print('child generation:', generation + 1)
        print('genome_id:', genome_id)
        # init MyDriver
        my_driver = MyDriver()
        
        # create feed forward network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # pass net to my_driver for computing output during race
        my_driver.output_net = net

        # start to race
        main(my_driver)

        # race fitness        
        if fitness_factor['raced_distance'] < 0.000001:
            genome.fitness = -1000.0
        else:
            genome.fitness = fitness_factor['raced_distance'] * fitness_factor['average_speed']
        print('-------------------Race result-----------------------')
        print('Fitness:', genome.fitness)
        
        print('-------------------End race-----------------------') 

    print('-------------------Evolution end-----------------------')
    generation += 1

def run(config_file):
    global RETRAIN
    global TRAIN

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)


    if not TRAIN:
        # load the best winner
        with open('winner-feedforward', 'rb') as f:
            genome = pickle.load(f)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        my_driver = MyDriver()
        my_driver.output_net = net
        start = time.time()
        main(my_driver)
        end = time.time()
    else:

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        if RETRAIN:
            # if you want to continue the training, neat-cehckpoint-#, # means the number of generation
            p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-' + str(RETRAIN_FROM_GENERATION))

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))

        # Start to train
        winner = p.run(eval_genomes)

        # save the best winner network
        with open('winner-feedforward', 'wb') as f:
            pickle.dump(winner, f)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        
        # For reference
        '''node_names = {-1:'A', -2: 'B', 0:'A XOR B'}'''
        '''visualize.draw_net(config, winner, True, node_names=node_names)'''
        
        # visualize training result
        visualize.draw_net(config, winner, True)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)


# main
report = neat.StdOutReporter(True)
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
run(config_path)


