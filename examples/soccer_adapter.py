"""Drive the Robot Soccer Kit base 25 cm from its starting position."""
import math


class Drive:
    def __init__(self, model, data):
        self.position = data.joint("base/base_freejoint").qpos[:2]
        self.start = self.position.copy()
        self.wheels = [data.actuator(f"base/wheel{i}_speed").ctrl for i in (1, 2, 3)]

    def before_step(self):
        speed = 8.0 if math.dist(self.start, self.position) < 0.25 else 0.0
        for wheel, target in zip(self.wheels, (speed, -speed, 0.0)):
            wheel[0] = target

    def close(self):
        for wheel in self.wheels:
            wheel[0] = 0.0
