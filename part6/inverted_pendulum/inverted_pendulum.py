import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class InvertedPendulum:
    def __init__(self,pole_mass=2.0,cart_mass=8.0,pole_length=0.5,delta_t=0.1):
        self.angle_list=list()
        self.pole_mass=pole_mass
        self.cart_mass=cart_mass
        self.pole_length=pole_length

        self.gravity=9.8
        
        self.delta_t=delta_t
        self.angle_t=np.random.normal(0,0.05)
        self.angular_velocity_t=0.0
        self.alpha = 1.0/(self.pole_mass+self.cart_mass)

    def reset(self, exploring_starts=True, initial_angle=-0.5):
        if exploring_starts:
            initial_angle= np.random.uniform(0,np.pi/2.0)
        
 

        self.angle_list =[]
        self.angle_t=np.random.uniform(0,0.1)
        self.angular_velocity_t=0.0
        self.angle_list.append(initial_angle)
        return [self.angle_t,self.angular_velocity_t]

    def step(self,action):
        if(action>=3):
            raise ValueError("The action value is out of range")

        done=False
        reward = -0.01
        action_list=[-50,0,50]
        action_t=action_list[action]
        angular_velocity_t1 = self.angular_velocity_t + \
                              (self.gravity * np.sin(self.angle_t) - \
                              self.alpha * self.pole_mass * self.pole_length * np.power(self.angular_velocity_t, 2) * (np.sin(2*self.angular_velocity_t)/2.0) + \
                              self.alpha * np.cos(self.angle_t) * action_t) / \
                              ((4/3) * self.pole_length - self.alpha * self.pole_mass * self.pole_length * np.power(np.sin(self.angle_t), 2)) * self.delta_t

        angle_t1=self.angle_t+angular_velocity_t1*self.delta_t

        if angle_t1 < -(np.pi/2.0):
            angle_t1 = -(np.pi/2.0)
            angular_velocity_t1 = 0
        if angle_t1 > (np.pi/2.0):
            angle_t1 = (np.pi/2.0)
            angular_velocity_t1 = 0

        self.angle_t=angle_t1
        self.angular_velocity_t=angular_velocity_t1
        self.angle_list.append(angle_t1)

        if angle_t1 >= (np.pi/2.0) or angle_t1 <= -(np.pi/2.0):
            reward = 0.0
            done = True
        else:
            reward = np.cos(angle_t1)
            done = False    

        return [angle_t1,angular_velocity_t1] , reward, done

    def render(self, file_path='./inverted_pendulum.mp4', mode='mp4'):
        """ When the method is called it saves an animation
        of the steps happened until that point in the episode.
        Ideally it should be called at the end of the episode,
        or every k episodes.
        
        ATTENTION: It requires avconv and/or imagemagick installed.
        @param file_path: the name and path of the video file
        @param mode: the file can be saved as 'gif' or 'mp4'
        """
        # Plot init
        fig = plt.figure()
        axis_limit = self.pole_length + (self.pole_length * 0.5)
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-axis_limit, axis_limit), ylim=(0.0, 1.5*axis_limit))
        ax.grid(False)  # disable the grid
        ax.set_aspect('equal')
        ax.set_yticklabels([])
        # x_line = np.linspace(start=-axis_limit, stop=axis_limit, num=100)
        # y_line = np.zeros(100)
        # ax.plot(x_line, y_line)  # plot the base-line
        # line, _ = ax.plot(x, y, 'o-', lw=2)
        line, = ax.plot([], [],color='black', linestyle='solid', linewidth=1.5, marker='o', markerfacecolor='#aa0000', markersize=10, zorder=1)
        # Adding the brown circle pad
        circle = plt.Circle((0.0,-0.01), radius=0.05, color='#2b2200', fill=True, zorder=2)
        ax.add_patch(circle)
        # Adding the text
        time_text = ax.text(0.05, 0.85, '', transform=ax.transAxes)
        _angle_list = self.angle_list
        _delta_t = self.delta_t

        def _init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def _animate(i):
            angle_cos = np.cos(_angle_list[i]) * self.pole_length
            angle_sin = np.sin(_angle_list[i]) * self.pole_length
            x1, y1 = [0, angle_sin], [0, angle_cos]
            #y1 = (angle_cos, angle_sin)
            line.set_data(x1, y1)
            time_text.set_text("Time: " + str(np.round(i*_delta_t, 1)) + "s" + '\n' + "Frame: " + str(i))
            return line, time_text

        ani = animation.FuncAnimation(fig, _animate, np.arange(1, len(self.angle_list)),
                                      blit=True, init_func=_init, repeat=False)

        if mode == 'gif':
            ani.save(file_path, writer='imagemagick', fps=int(1/self.delta_t))
        elif mode == 'mp4':
            ani.save(file_path, fps=int(1/self.delta_t), writer='avconv', codec='libx264')
        # Clear the figure
        fig.clear()
        plt.close(fig)
