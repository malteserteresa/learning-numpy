"""
The equation for the trajectory of a ball can be solved to find the time it takes the ball to reach a certain height
How long does it take for the ball to reach 0.2m with an initial velocity of 5m per second?
"""
import math

v0 = 5.0
g = 9.81
y = 0.2

t1 = v0 - math.sqrt((v0**2 - 2*g*y))

t2 = v0 + math.sqrt((v0**2 - 2*g*y))


print(f"At t={t1} s and {t2} s, the height is 0.2 m")

