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

TRAIN = True
RETRAIN = False
DUMP_WINNER = False
SWARM = False

if TRAIN == False and RETRAIN == True:
    print('TRAIN is False, but RETRAIN is True!')
    exit(0)

RETRAIN_FROM_GENERATION = 33
generation = 0
if RETRAIN:
    generation = RETRAIN_FROM_GENERATION

DEBUG = False
VERBOSE = True
ACC = False
ACC_value = 7


# global variable
fitness_factor = {}
termination = False
count_s = 0

SWARM_FILE = 'group_19_swarm_sync.txt'
CAR_FILE_A = 'group_19_swarm_car.txt'
CAR_FILE_B = 'group_19_swarm_carb.txt'
CAR2_FILE_A = 'group_19_swarm_car2.txt'
CAR2_FILE_B = 'group_19_swarm_car2b.txt'

CAR_ID = 1
CAR2_ID = 2    

f = open(SWARM_FILE, 'r')
car1_start = False
for line in f:
    print(line)
    if 'car1_start' in line:
        car1_start = True
f.close()
if car1_start:
    CAR_ID = 2
    CAR2_ID = 1
    open(SWARM_FILE, 'w').close()
else:
    f = open(SWARM_FILE, 'w')
    f.write('car1_start')
    f.close()

import pyautogui
pyautogui.FAILSAFE = False
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

_thread.start_new_thread(switch_ACC, ('Thread1', ))

def tanh(x):
    x = max(-60.0, min(60.0, 2.5 * x))
    return math.tanh(0.3 * x)

class Swarm:    
    def __init__(self):
        pass

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
        
        if SWARM:
            if count_s > 20:
                f2 = None
                if count_s % 10 == 0:
                    f2 = open(CAR_FILE_B, 'r')
                elif count_s % 10 == 5:
                    f2 = open(CAR_FILE_A, 'r')
                if f2 != None:
                    pass
                    #print(f2.readline())

        global accerlate_credit
        global break_penalty
        global last_steering 
        global stable_penalty
        global speed_reward

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

        if not hasattr(self, 'volatility'):
            self.volatility = 0.0

        if not hasattr(self, 'volatility'):
            self.volatility = 0.0

        if not hasattr(self, 'average_angle'):
            self.average_angle = 0.0
        
        if not hasattr(self, 'speed_reward'):
            self.speed_reward = 0.0

        self.speed_reward += speed*math.cos(math.radians(carstate.angle))

        self.average_speed = (self.average_speed * (count_s - 1) + speed) / float(count_s)

        self.average_angle = (self.average_angle * (count_s - 1) + np.cos(math.radians(carstate.angle))) / float(count_s)


        if not hasattr(self, 'prev_steering'):
            self.prev_steering = 0.0
        
        if not hasattr(self, 'accumulate_speed_multiple_cos_angle'):
            self.accumulate_speed_multiple_cos_angle = 0.0

        if not hasattr(self, 'never_use_steer'):
            self.never_use_steer = True
        
        # prepare x
        #x = [carstate.speed_x * 3.6, carstate.distance_from_center, carstate.angle] + list(carstate.distances_from_edge)
        x = [(carstate.speed_x * 3.6 - 100) / 200, carstate.distance_from_center, carstate.angle / 360] + ((np.array(list(carstate.distances_from_edge)) - 100) / 200).tolist() + ((np.array(list(carstate.opponents))[15:20] - 100) / 200).tolist()
        #x = [(carstate.speed_x * 3.6 - 100) / 200, carstate.distance_from_center, carstate.angle / 360] + ((np.array(list(carstate.distances_from_edge)) - 100) / 200).tolist()
        #print(carstate.distances_from_edge, carstate.angle)
        self.accumulate_speed_multiple_cos_angle += speed * np.cos(math.radians(carstate.angle))

        #print(carstate.angle / 180 * 3.14)

        if DEBUG:
            print('x', x)

        # normalize x
        '''x = tc.scaler.transform([x])'''

        # create output
        y = self.output_net.activate(x)

        self.volatility = (self.volatility * (count_s - 1) + 5 * abs(y[2] - self.prev_steering)) / float(count_s)

        self.prev_steering = y[2]

        if DEBUG:
            print('y', y)
        
        # assign output
        if SWARM:
            if count_s % 10 == 0:
                f1 = open(CAR_FILE_A, 'w+')
                f1.write(str(speed))
                f1.close()
            elif count_s % 10 == 5:
                f1 = open(CAR_FILE_B, 'w+')
                f1.write(str(speed))
                f1.close()

        if abs(y[2]) > 0.1 and speed > 1:
            self.never_use_steer = False

        # automatically switch gear
        if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1
        
        '''min_value = 199.0
        closest_idx = -1
        for idx, value in enumerate(carstate.opponents):
            if value < min_value:
                min_value = value
                closest_idx = idx
        
        if closest_idx >= 0:
            if carstate.opponents[17] < 100 or carstate.opponents[16] < 100 or \
                    carstate.opponents[18] < 100:
                if carstate.distance_from_center >= 0.0 and carstate.distance_from_center < 1:
                    self.steer(carstate, -0.5, command)
                elif carstate.distance_from_center > -1 and carstate.distance_from_center < -0.0:
                    self.steer(carstate, 0.5, command)

            else:
                self.steer(carstate, 0.0, command)'''
        #elif carstate.opponents[17] < 100 or carstate.opponents[16] < 50:
        #    self.steer(carstate, -0.5, command)
        #else:
        #    self.steer(carstate, -0.5, command)

        if speed < 150 and carstate.distance_raced < 60:
            self.steer(carstate, 0.0, command)
            self.accelerate(carstate, 150, command)
            
        elif carstate.distance_from_center >= 1 or carstate.distance_from_center <= -1:
            self.steer(carstate, 0.0, command)
            self.accelerate(carstate, 80, command)
        else:
            command.accelerator = y[0]
            command.brake = y[1]
            command.steering = y[2]

        # Termination condition
        # 1. if car speed < 10 after count_s > 100 
        # 2. hit wall
        # 3. off track
        if TRAIN or DUMP_WINNER:
            if (carstate.current_lap_time > 3 and abs(speed - 0.0) < 2) or \
               carstate.damage > 1000 or \
               carstate.distance_from_center >= 1 or \
               carstate.distance_from_center <= -1 or \
               carstate.distance_raced > 7777:
            

                print('Termination!!!!!!!!!!!!!!')
                termination = True
                
            #if ACC and not hasattr(self, 'ACC_flag'):
            #    self.ACC_flag = True
                #press_acc()               

            if termination:
                
                # fitness
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
                print(fitness_factor['speed_reward'], fitness_factor['accumulate_speed_multiple_cos_angle'])
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
        
def eval_genomes(genomes, config):
    global fitness_factor
    global generation
    global DUMP_WINNER
    global ACC
    print('parent generation:', generation)
    print('-------------------Evolution start-----------------------')
    #test = False
    for genome_id, genome in genomes:
        #time.sleep(2)
        print('-------------------Start to race-----------------------')
        print('child generation:', generation + 1)
        print('genome_id:', genome_id)
        # init MyDriver
        #if test:
        #    genome.fitness = 1
        #    continue
        my_driver = MyDriver()
        
        # create feed forward network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # pass net to my_driver for computing output during race
        my_driver.output_net = net

        if ACC:
            press_acc()
        # start to race
        main(my_driver)

        # race fitness        
        if fitness_factor['raced_distance'] < 0.000001 or fitness_factor['never_use_steer']:
        #if fitness_factor['raced_distance'] < 0.000001:
            genome.fitness = -1000.0
        else:
            genome.fitness = fitness_factor['raced_distance'] * fitness_factor['average_speed'] * fitness_factor['average_angle']
            #genome.fitness = fitness_factor['raced_distance'] * fitness_factor['average_speed'] * fitness_factor['accumulate_speed_multiple_cos_angle']
            #genome.fitness = fitness_factor['raced_distance'] / 10 + fitness_factor['average_speed'] 
            #genome.fitness = fitness_factor['accumulate_speed_multiple_cos_angle']
            #genome.fitness = fitness_factor['speed_reward']*0.06 + fitness_factor['raced_distance']*2- fitness_factor['damage']*100 -fitness_factor['time']

            #genome.fitness = fitness_factor['accumulate_speed_multiple_cos_angle'] * math.log(fitness_factor['average_speed'] + 1) / (math.log(fitness_factor['damage'] + 1) + 1)



        print('-------------------Race result-----------------------')
        print('Fitness:', genome.fitness)
        #if not test:
        #    test = True
        print('-------------------End race-----------------------') 

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
            winner = p.run(eval_genomes)

            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(5))

            visualize.draw_net(config, winner, True)
            
            # Display the winning genome.
            print('\nBest genome:\n{!s}'.format(winner))

            #visualize.plot_stats(stats, ylog=False, view=True)
            #visualize.plot_species(stats, view=True)

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


