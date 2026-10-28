from inverted_pendulum import InvertedPendulum
import random
my_pole=InvertedPendulum(pole_mass=2.0, cart_mass=8.0, pole_length=0.5, delta_t=0.1)
cumulated_reward=0

for step in range(100):
    action=random.randint(0,2)
    observation,reward,done=my_pole.step(action)
    cumulated_reward+=reward
    print("Step: " + str(step))
    print("Action: " + str(action))
    print("Angle: " + str(observation[0]))
    print("Velocity: " + str(observation[1]))
    print("Reward: " + str(reward))
    print("")
    if done: break

print("Finished after: " + str(step+1) + " steps")
print("Cumulated Reward: " + str(cumulated_reward))
print("Saving the gif in: ./inverted_pendulum.gif")
my_pole.render(file_path='./inverted_pendulum.gif', mode='gif')
print("Complete!")