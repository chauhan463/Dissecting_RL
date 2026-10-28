from mountain_car import MountainCar
import random
my_car=MountainCar(mass=0.2,friction=0.3,delta_t=0.1)
cumulated_reward=0
my_car.reset()
for step in range(100):
    action=random.randint(0,2)
    observation,reward,done=my_car.step(action)
    cumulated_reward+=reward
    if done: break

print("Finished after: " + str(step+1) + " steps")
print("Cumulated Reward: " + str(cumulated_reward))
print("Saving the gif in: ./mountain_car.gif")
my_car.render(file_path='./mountain_car.gif', mode='gif')
print("Complete!")