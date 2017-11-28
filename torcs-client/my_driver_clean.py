from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np
import math

def activation_function(z):
    #return 1 / (1 + np.exp(-x))
    z = max(-60.0, min(60.0, 2.5 * z))
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

f = open('winner_result.txt', 'r')
for line in f:
    if 'DefaultNodeGene' in line:
        node_name = find_between(line, 'key=', ', ')
        node_bias = float(find_between(line, 'bias=', ', '))
        
        if node_name not in node_dict.node_dict:
            node_dict.add(Node(node_name, node_bias, [], []))
        
    elif 'DefaultConnectionGene' in line:
        node_name = find_between(line, 'key=(', '), ').split(', ')
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

class MyDriver(Driver):
    
    def drive(self, carstate: State) -> Command:
        x = [carstate.speed_x * 3.6, carstate.distance_from_center, carstate.angle] + list(carstate.distances_from_edge)
        for x_idx in range(22):
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
        #print(y)
        return command
