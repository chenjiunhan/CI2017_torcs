#! /usr/bin/env python3
from __future__ import print_function
from pytocl.main import main
import _thread
import time
import socket

# exchange information between cars
car1_str = None
car2_str = None
SOCKET = False
# check if 2 cars come in game.
token = [1, 2]
addrs = []
clients = []


# exchange through socket
def socket_thread(name, s):
    while True:
        c, addr = s.accept()
        _thread.start_new_thread(swarm_thread, (c, addr))

def swarm_thread(c, addr):  
    global car1_str
    global car2_str
    global token
    global addrs
    global clients
    
    while True:                
        if len(token) > 0:
            if addr not in addrs:
                addrs.append(addr)
                send_str = 'token:' + str(token.pop())
                c.send(send_str.encode())
                print('Got connection from', addr)
            else:
                send_str = 'already in'
                print(send_str)
                ack = c.recv(128).decode()
                if ack == 'start':
                    if len(token) == 0:            
                        print('exit00000')
                        break  
                    if len(token) > 0:
                        send_str = 'exit'
                        c.send(send_str.encode())
                        exit(0)
                                    
        if len(token) == 0:            
            break
        

    while True:
        
        
        car_str = c.recv(128).decode()
        if car_str == 'start':
            continue
        
        car_str_split = car_str.split(',')
        if len(car_str_split) <= 1:
            continue

        car_id = int(car_str_split[0])
        car_speed = float(car_str_split[1])        
        race_position = int(car_str_split[2])
        distance_raced = float(car_str_split[3])
        
        if car_id == 1:    
            car1_str = car_str      
            if car2_str != None:
                c.send(car2_str.encode())
                #print('server send1', car_str)
            else:
                c.send(b'not ready')
        else:
            car2_str = car_str
            if car1_str != None:
                c.send(car1_str.encode())
                #print('server send2', car_str)
            else:
                c.send(b'not ready')
        if not car_str:
            break
       

 
    c.close()       
   
if __name__ == '__main__':
    print('group_19_v0.91')
    if SOCKET:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
            print("Socket successfully created")
            port = 52230
            s.bind(('', port))        
            print("socket binded to %s" %(port))
            s.listen(5)     
            print("socket is listening")
        
            _thread.start_new_thread(socket_thread, ('socket_thread', s))
        except OSError as e:
            print('socket has some OS problems.')
            print(e)
            s.close()
        except e:
            print('socket has some problems.')
            print(e)
            s.close()
        time.sleep(1)
    
    import my_driver_clean 
    main(my_driver_clean.MyDriver(logdata=False))
    
    if SOCKET:
        try:
            s.close() 
        except:
            pass
    
