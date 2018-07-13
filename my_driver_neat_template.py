from __future__ import print_function
from pytocl.main import main
from pytocl.driver import Driver
from pytocl.car import State, Command
import pytocl.main
import math
import _thread

import numpy as np
import os
import pickle

import neat
import visualize

import time

#############################################################

# This is a driver for training, not for running on server.
# The winner weights will be save as winner_result#.txt

#############################################################


TRAIN = True
RETRAIN = True
DUMP_WINNER = False
SWARM = False

if TRAIN == False and (RETRAIN == True and DUMP_WINNER == False):
    print('TRAIN is False, but RETRAIN is True!')
    exit(0)

RETRAIN_FROM_GENERATION = 370
generation = 0
if RETRAIN:
    generation = RETRAIN_FROM_GENERATION

DEBUG = False
VERBOSE = True

# Accelerate traning
ACC = False
ACC_value = 7

# global variable
fitness_factor = {}
termination = False
count_s = 0

# for swarm
car_write_stack = []
car_read = None

# for simulate user input to accelerate training
import pyautogui
pyautogui.FAILSAFE = False
def press_acc():
    for i in range(ACC_value):
        pyautogui.press('+')

def switch_ACC(name):    
    global ACC
    while True:
        user_input = input()
        print(user_input)
        if user_input == 'on':
            ACC = True
            time.sleep(1)
            press_acc()
        elif user_input == 'off':
            ACC = False

# type 'on' in terminal to accelerate, type 'off' to stop.
_thread.start_new_thread(switch_ACC, ('Thread1', ))

def tanh(x):
    return math.tanh(x)

class MyDriver(Driver):    
    def drive(self, carstate: State) -> Command:
               
        # count from race start
        global count_s
        global ACC        
        global client
        global TRAIN
        # if race finished
        global termination                

        global CAR_ID
        global CAR2_ID
        global CAR_FILE_A
        global CAR_FILE_B

        global CAR2_FILE_A
        global CAR2_FILE_B
        
        global accerlate_credit
        global break_penalty
        global last_steering 
        global stable_penalty
        global speed_reward

        command = Command()
        self.current_accelerator_controller = 'NEAT'
        self.current_break_controller = 'NEAT'
        self.current_steer_controller = 'NEAT'

        self.swing = False
        self.closer = False
       
        count_s += 1
        
        # * 3.6 to KM/H
        speed = carstate.speed_x * 3.6
 
        min_edge_value = 199.0
        closest_edge_idx = -1
        for idx, value in enumerate(carstate.distances_from_edge):
            if value < min_edge_value:
                min_edge_value = value
                closest_edge_idx = idx
        

        # calculate top speed
        if not hasattr(self, 'top_speed'):
            self.top_speed = 0.0

        if speed > self.top_speed:
            self.top_speed = speed
        
        if not hasattr(self, 'stuck'):
            self.stuck = False
            self.stuck_count = 0
        
        # calculate average speed
        if not hasattr(self, 'average_speed'):
            self.average_speed = 0.0

        if not hasattr(self, 'volatility'):
            self.volatility = 0.0

        if not hasattr(self, 'volatility'):
            self.volatility = 0.0

        if not hasattr(self, 'average_angle'):
            self.average_angle = 0.0
        
        if not hasattr(self, 'speed_reward'):
            self.speed_reward = 0.0
        
        # some candidate of fitness funtion factors
        self.speed_reward += speed*math.cos(math.radians(carstate.angle))
        self.average_speed = (self.average_speed * (count_s - 1) + speed) / float(count_s)
        self.average_angle = (self.average_angle * (count_s - 1) + np.cos(math.radians(carstate.angle))) / float(count_s)

        if not hasattr(self, 'prev_steering'):
            self.prev_steering = 0.0
        
        if not hasattr(self, 'accumulate_speed_multiple_cos_angle'):
            self.accumulate_speed_multiple_cos_angle = 0.0

        if not hasattr(self, 'never_use_steer'):
            self.never_use_steer = True
        
        # input features
        if carstate.speed_x < 1:
            speed_x = 1
        else: 
            speed_x = carstate.speed_x

        radio_speed_xy = carstate.speed_z / speed_x
        x = [(carstate.speed_x * 3.6 - 100) / 200, carstate.distance_from_center, carstate.angle / 360, radio_speed_xy] + ((np.array(list(carstate.distances_from_edge)) - 100) / 200).tolist()

        self.accumulate_speed_multiple_cos_angle += speed * np.cos(math.radians(carstate.angle))

        if DEBUG:
            print('x', x)

        # create output
        y = self.output_net.activate(x)

        # some candidate of fitness funtion factors
        self.volatility = (self.volatility * (count_s - 1) + 5 * abs(y[2] - self.prev_steering)) / float(count_s)
        self.prev_steering = y[2]

        if DEBUG:
            print('y', y)
        
        # some candidate of fitness funtion factors 
        if abs(y[2]) > 0.1 and speed > 1:
            self.never_use_steer = False
        
        # automatically switch gear
        if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1
        
        # assign output        
        command.accelerator = y[0]
        command.brake = y[1]
        command.steering = y[2]            

        # Termination condition
        # 1. speed too low for long time.
        # 2. hit wall
        # 3. off track
        # 4. run long enough

        if TRAIN or DUMP_WINNER:
            if (carstate.current_lap_time > 3 and abs(speed - 0.0) < 3) or \
               (carstate.current_lap_time > 30 and abs(speed - 0.0) < 20) or \
               carstate.damage > 0 or \
               carstate.distance_from_center >= 1 or \
               carstate.distance_from_center <= -1 or \
               carstate.distance_raced > 13000:
               
                termination = True
                
            if termination:
                
                # fitness factors
                fitness_factor['raced_distance'] = carstate.distance_raced
                fitness_factor['top_speed'] = self.top_speed
                fitness_factor['average_speed'] = self.average_speed
                fitness_factor['volatility'] = self.volatility
                fitness_factor['average_angle'] = self.average_angle
                fitness_factor['accumulate_speed_multiple_cos_angle'] = self.accumulate_speed_multiple_cos_angle
                fitness_factor['damage'] = carstate.damage
                fitness_factor['never_use_steer'] = self.never_use_steer
                fitness_factor['speed_reward'] = self.speed_reward
                fitness_factor['time'] = carstate.current_lap_time
                fitness_factor['last_lap_time'] = carstate.last_lap_time
                
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
        
def eval_genomes(genomes, config):
    global fitness_factor
    global generation
    global DUMP_WINNER
    global ACC
    print('parent generation:', generation)
    print('-------------------Evolution start-----------------------')
    
    record_f = open('record_fitness.txt', 'a+')
    record_f.write('generation:' + str(generation) + '\n')
    for genome_id, genome in genomes:
        #time.sleep(2)
        print('-------------------Start to race-----------------------')
        print('child generation:', generation + 1)
        print('genome_id:', genome_id)

        #if genome_id == 40:
        '''if genome_id == 1750:
            genome.fitness = 1000000
        else:
            genome.fitness = 0
            continue'''
        
        my_driver = MyDriver()
        
        # create feed forward network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # pass net to my_driver for computing output during race
        my_driver.output_net = net

        # accelerate training
        if ACC:
            press_acc()
        
        # start to race
        main(my_driver)

        # race fitness        
        if fitness_factor['raced_distance'] < 0.000001 or fitness_factor['never_use_steer']:
            genome.fitness = -1000.0
        else:
            genome.fitness = fitness_factor['raced_distance'] * fitness_factor['average_speed'] * fitness_factor['average_angle']
            
            # for drawing graph
            fitness_value = fitness_factor['raced_distance'] * fitness_factor['average_speed'] * fitness_factor['average_angle']            
            record_f.write(str(fitness_value) + ',' + str(fitness_factor['raced_distance']) + ',' + str(fitness_factor['average_speed']) + ',' + str(fitness_factor['average_angle']) + ',' + str(fitness_factor['last_lap_time']) + '\n')
            
            '''
            #genome.fitness = fitness_factor['raced_distance'] * fitness_factor['average_speed'] * fitness_factor['accumulate_speed_multiple_cos_angle']
            #genome.fitness = fitness_factor['raced_distance'] / 10 + fitness_factor['average_speed'] 
            #genome.fitness = fitness_factor['accumulate_speed_multiple_cos_angle']
            #genome.fitness = fitness_factor['speed_reward']*0.06 + fitness_factor['raced_distance']*2- fitness_factor['damage']*100 -fitness_factor['time']
            #genome.fitness = math.log(fitness_factor['raced_distance']) * fitness_factor['accumulate_speed_multiple_cos_angle'] * math.log(fitness_factor['average_speed'] + 1)
            '''

        print('-------------------Race result-----------------------')
        print('Fitness:', genome.fitness)
        print('-------------------End race-----------------------') 
    record_f.close()
    print('-------------------Evolution end-----------------------')
    generation += 1
    
    


def run(config_file):
    global RETRAIN
    global TRAIN
    global DUMP_WINNER

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    config.genome_config.add_activation('tanh_function', tanh)

    if not TRAIN:
        # load the best winner
        if DUMP_WINNER:
            p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-' + str(RETRAIN_FROM_GENERATION))
            # Start to train
            winner = p.run(eval_genomes, 1)

            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(5))

            visualize.draw_net(config, winner, True)
            
            # Display the winning genome.
            print('\nBest genome:\n{!s}'.format(winner))

        # save the best winner network
            with open('winner-feedforward', 'wb') as f:
                pickle.dump(winner, f)

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
        winner = p.run(eval_genomes, 95)

        # save the best winner network
        with open('winner-feedforward', 'wb') as f:
            pickle.dump(winner, f)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        
        # visualize training result
        visualize.draw_net(config, winner, True)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)


# main
report = neat.StdOutReporter(True)
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
run(config_path)


