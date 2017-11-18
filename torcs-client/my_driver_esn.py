from pytocl.driver import Driver
from pytocl.car import State, Command
import trainer2_chen as tc
import numpy as np

class MyDriver(Driver):
    def drive(self, carstate: State) -> Command:
        command = Command()
        x = [carstate.speed_x * 3.6, carstate.distance_from_center, carstate.angle] + list(carstate.distances_from_edge)
        #x = tc.scaler.transform([x])
        x = np.array([x]).reshape(1, len(x))
        print(x)
        y = tc.esn.predict(x)
        y = y[0]
        print(x)
        print(y)
        #command.accelerator = y[0]
        command.accelerator = 1
        #command.brake = y[1]
        command.brake = 0
        command.steering = y[2]

        if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1        
        command.gear = 1
        #if command.brake < 0.5 or carstate.speed_x * 3.6 < 30:
        #    command.brake = 0
        #if carstate.speed_x < 50:
        #    command.brake = 0
        #self.steer(carstate, 0.0, command)
        
        return command
