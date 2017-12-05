from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np
import math
import _thread


#import neat
import socket  
#import os
#import pickle
#import time
#from pytocl.main import main

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
car_str = None
friend_car_str = None

# Define the port on which you want to connect
port = 55229                

# connect to the server on local computer
s.connect(('127.0.0.1', port))
s.setblocking(True)

data = s.recv(128)

SWARM = True

TOTAL_STUCK_LIMIT = 20

print(data.decode().strip().split(':')[1])

CAR_ID = int(data.decode().strip().split(':')[1])

if CAR_ID == 1:
    FRIEND_CAR_ID = 2
elif CAR_ID == 2:
    FRIEND_CAR_ID = 1    
else:
    print("CAR_ID Error")
    raise 

def socket_client_thread(name, s):
    global car_str
    global friend_car_str
    while True:                
        #print('recv friend')
        friend_car_str = s.recv(128)
        #print('friend_car_str', friend_car_str)
#_thread.start_new_thread(socket_client_thread, ('socket_client_thread', s))


def activation_function(z):
    #return 1 / (1 + np.exp(-x))
    #z = max(-60.0, min(60.0, 2.5 * z))
    return math.tanh(z)

class Node_dict:
    def __init__(self):
        self.node_dict = {}
        self.available_node_dict = {}

    def add(self, node):
        self.node_dict[node.name] = node

    def reset(self):
        for key, node in self.node_dict.items():
            node.reset()
class Node:    
    def __init__(self, name, bias, target_weight, expect_input, finished = False, value = 0.0):
        self.name = str(name)
        self.bias = bias
        self.target_weight = target_weight
        self.expect_input = expect_input
        self.value = value
        self.finished = finished

    def output(self):
        return activation_function(self.value)

    def __hash__(self):
        return hash(self.name)

    def reset(self):
        self.value = 0.0
        if self.expect_input != None:
            self.finished = False
def forward():
    global node_dict
    while True:    
        
        all_node_ready = True

        for key, node in node_dict.node_dict.items():

            value = 0.0
            x = []
            #print('------------------')
            if node.finished == False:
                all_node_ready = False
            else:
                continue

            if node.expect_input == None:
                continue
            all_input_ready = True
            for expect_input in node.expect_input:        
                           
                if node_dict.node_dict[str(expect_input)].finished == False:
                    all_input_ready = False
                    break

                for target_weight in node_dict.node_dict[str(expect_input)].target_weight:
                    if str(target_weight[0]) == str(node.name):
                        x.append(target_weight[1] * node_dict.node_dict[str(expect_input)].value)
                        value += target_weight[1] * node_dict.node_dict[str(expect_input)].value
                        #print('value += ', target_weight[1], '*', node_dict.node_dict[str(expect_input)].value)

            if all_input_ready:
                #print(key, 'ok')
                node.finished = True
                node.value += node.bias
                #print('bias', node.bias)
                node.value += sum(x)
                #print('v1', node.value)
                node.value = activation_function(node.value)
                #print(key, node.value)
        
        if all_node_ready:
            break

    return node_dict.node_dict['0'].value, node_dict.node_dict['1'].value, node_dict.node_dict['2'].value
#    return 

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

node_dict = Node_dict()

f = open('winner_result3.txt', 'r')
for line in f:
    if 'DefaultNodeGene' in line:
        node_name = find_between(line, 'key=', ', ')
        node_bias = float(find_between(line, 'bias=', ', '))
        
        if node_name not in node_dict.node_dict:
            node_dict.add(Node(node_name, node_bias, [], []))
        
    elif 'DefaultConnectionGene' in line:
        node_name = find_between(line, 'key=(', '), ').split(', ')
        node_enabled = find_between(line, 'enabled=', ')')
        if node_enabled == 'False':
            continue
        node_weight = float(find_between(line, 'weight=', ', '))
        if node_name[0] not in node_dict.node_dict: #input
            node_dict.add(Node(node_name[0], None, [], None, True))
    
        node_dict.node_dict[node_name[0]].target_weight.append((node_name[1], node_weight))
        node_dict.node_dict[node_name[1]].expect_input.append(node_name[0])



'''node_dict.add(Node(-22, None, [(1, )], None, True))
node_dict.add(Node(-21, None, [(0, ), (1, ), (2, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
node_dict.add(Node(-22, None, [(, )], None, True))
'''
for key, node in node_dict.node_dict.items():
    print(node.name, node.bias, node.target_weight, node.expect_input)


count_s = 0
class MyDriver(Driver):
    
    def drive(self, carstate: State) -> Command:         
        global car_str
        global friend_car_str
        global s
        global count_s
        global SWARM
        global TOTAL_STUCK_LIMIT
        
        if not hasattr(self, 'stuck'):
            self.stuck = False
            self.stuck_count = 0
            self.recover_stuck = False
        if not hasattr(self, 'reverse'):
            self.reverse = False

        if not hasattr(self, 'total_stuck'):
            self.total_stuck = 0
        
        if count_s == 0:
            s.send(b'start')

        count_s += 1
    
        speed = carstate.speed_x * 3.6       
        if carstate.speed_x < 1:
            speed_x = 1
        else: 
            speed_x = carstate.speed_x

        radio_speed_xy = carstate.speed_z / speed_x
        x = [(carstate.speed_x * 3.6 - 100) / 200, carstate.distance_from_center, carstate.angle / 360, radio_speed_xy] + ((np.array(list(carstate.distances_from_edge)) - 100) / 200).tolist()
        for x_idx in range(23):
            if str(-(x_idx + 1)) in node_dict.node_dict:
                node_dict.node_dict[str(-(x_idx + 1))].value = x[x_idx]

        y = forward()
        command = Command()
        command.accelerator = y[0]
        command.brake = y[1]
        command.steering = y[2]

        node_dict.reset()
        if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1
        if abs(command.steering) <= 0.8:
            if (np.where(((np.array(list(carstate.opponents[0:13]) + list(carstate.opponents[23:36])))) <= 100)[0]).size > 0:
                #print('c1', command.brake)
                command.accelerator += 1.0
                command.brake -= 0.3
                #pass
                #print('c2', command.brake)
                #self.accelerate(carstate, speed + 50, command)
                print('oppoent behind you, accelarate! ')
            elif (np.where(((np.array(list(carstate.opponents[14:18]) + list(carstate.opponents[20:23])))) <= 100)[0]).size > 0:
                #print('c1', command.brake)
                command.accelerator += 1.0
                command.brake -= 0.3
                #pass
                #print('c2', command.brake)
                #self.accelerate(carstate, speed + 50, command)
                print('oppoent in front of you, accelarate! ')
            
            elif (np.where(((np.array(list(carstate.opponents[18:19])))) < 50 )[0]).size > 0:
                #print('s1', command.steering)
                if carstate.distance_from_center < 0.2 and carstate.distance_from_center >= 0:
                    #print('s3')
                    command.steering = command.steering - 3.0
                elif carstate.distance_from_center > -0.2 and carstate.distance_from_center < 0:
                    #print('s4')
                    command.steering = command.steering + 3.0
                else:
                    #print('s5')
                    command.steering = command.steering + carstate.distance_from_center*-6.0
                #print('s2', command.steering)'''
        if SWARM:
            #print('update_car_str')
            car_str = str(CAR_ID) + ',' + str(round(speed, 2)) + "," + str(carstate.race_position) + "," + str(round(carstate.distance_raced, 2))
            if car_str != None:    
                s.send(car_str.encode())
                #print('send', car_str)
            
            friend_car_str = s.recv(128).decode()
            if friend_car_str == 'exit':
                SWARM = False                
            elif friend_car_str != '' and friend_car_str != 'not ready':
                friend_car_str_split = friend_car_str.split(',')
                friend_car_id = int(friend_car_str_split[0])
                friend_car_speed = float(friend_car_str_split[1])
                friend_race_position = int(friend_car_str_split[2])
                friend_distance_raced = float(friend_car_str_split[3])
                #print('race position', carstate.race_position, friend_race_position)
                if carstate.race_position > friend_race_position:
                    #print('diff', friend_distance_raced - carstate.distance_raced)
                    diff = friend_distance_raced - carstate.distance_raced
                    if diff < 0:
                        diff = 0
                        
                    if diff > 10 and diff < 50:
                        self.accelerate(carstate, friend_car_speed + diff / 5, command)
                        #print('closer', friend_car_speed)
                        
                    elif diff <= 10:
                        leave_diff = -(diff - 10)
                        self.accelerate(carstate, friend_car_speed - leave_diff * 2 , command)
                        #print('leave', friend_car_speed)
 
        if (carstate.angle > 170.0 or carstate.angle < -170) and speed > 20 and abs(carstate.distance_from_center) <= 0.7 and carstate.gear >= 1:
            self.reverse = True
        
        if carstate.angle < 30 and carstate.angle > -30 and abs(carstate.distance_from_center) <= 1:
            self.reverse = False

        if self.reverse:
            print('reverse')
            command.steering = np.sign(math.cos(carstate.angle)) * -1.0

        if self.stuck == False and carstate.distance_raced > 50 and speed < 2:
            self.stuck = True        
            self.total_stuck += 1

        #print(carstate.angle)
        #print(carstate.distance_from_center)
        #print(np.sign(math.cos(carstate.angle)) * 1.0)
        if self.total_stuck < TOTAL_STUCK_LIMIT:
            if abs(carstate.distance_from_center) >= 1 and not self.stuck:
                self.steer(carstate, 0.0, command)


        if self.stuck:
            print('stucked', self.stuck_count)
            if abs(carstate.distance_from_center) <= 0.5 or self.stuck_count > 500 or (carstate.angle < 30 and carstate.angle > -30):
                print('<<<<0.3')
                if carstate.gear <= -1:
                    print('0.3aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
                    command.gear = -1
                    command.brake = 1.0                    
                    command.accelerator = 0.0                    
                    if speed >= -0.1:
                        command.gear = 1
                elif carstate.gear >= 1:
                    print('<<<<0.3bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
                    command.gear = carstate.gear or 1
                    #command.accelerator = 1.0
                    #command.brake = 0.0

                    command.brake = 0.0
                    if abs(carstate.distance_from_center) >= 0.2:
                        print('default')
                        self.steer(carstate, 0.0, command)
                    else:
                        print('no stuck')
                        self.stuck = False
                        self.stuck_count = 0

                    if self.stuck_count > 1000:
                        self.stuck_count = 0

            elif carstate.distance_from_center <= -0.5:
                print('111111111111111')
                command.gear = -1
                command.accelerator = 0.5
                command.steering = -1.0
                command.brake = 0.0
                self.stuck_steer = command.steering
                
            elif carstate.distance_from_center >= 0.5:
                print('2222222222222222')
                command.gear = -1
                command.accelerator = 0.5
                command.steering = 1.0
                command.brake = 0.0
                self.stuck_steer = command.steering

            self.stuck_count += 1
        
        if not self.stuck and self.total_stuck >= TOTAL_STUCK_LIMIT:
            self.steer(carstate, 0.0, command)
            self.accelerate(carstate, 40, command)

        return command


