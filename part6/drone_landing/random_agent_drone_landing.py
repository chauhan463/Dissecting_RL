from drone_landing import DroneLanding
import numpy as np
my_drone=DroneLanding(world_size=11)
cumulated_reward=0

for step in range(50):
    action=np.random.randint(low=0,high=6)
    observation,reward,done=my_drone.step(action)
    cumulated_reward+=reward
    print("Action: " + str(action))
    print("x-y-z: " + str(observation))
    print("")

    if done: break

print("Finished after: " + str(step+1) + " steps")
print("Cumulated Reward: " + str(cumulated_reward))
my_drone.render(file_path='./random_drone_landing.gif', mode='gif')
print("Complete!")