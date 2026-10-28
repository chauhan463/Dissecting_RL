import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MountainCar:
    def __init__(self,mass=0.2,friction=0.3,delta_t=0.1):
        self.position_list=list()
        self.mass=mass
        self.gravity=9.8
        self.friction=friction
        self.delta_t=delta_t
        self.position_t=-0.5
        self.velocity_t=0.0

    def reset(self, exploring_starts=True, initial_position=-0.5):
        if exploring_starts:
            initial_position= np.random.uniform(-1.2,0.5)
        if initial_position < -1.2:
            initial_position=-1.2
        if initial_position > 0.5:
            initial_position = 0.5

        self.position_list =[]
        self.position_t=initial_position
        self.velocity_t=0.0
        self.position_list.append(initial_position)
        return [self.position_t,self.velocity_t]

    def step(self,action):
        if(action>=3):
            raise ValueError("The action value is out of range")

        done=False
        reward = -0.01
        action_list=[-0.2,0,0.2]
        action_t=action_list[action]
        velocity_t1=self.velocity_t +(   (-self.gravity*self.mass*np.cos(3*self.position_t)) + (action_t/self.mass) -(self.friction*self.velocity_t) )*self.delta_t
        position_t1=self.position_t+velocity_t1*self.delta_t

        if position_t1 <-1.2:
            position_t1 =-1.2
            velocity_t1=0

        self.position_t=position_t1
        self.velocity_t=velocity_t1
        self.position_list.append(position_t1)

        if position_t1>=0.5:
            reward = 1.0
            done =True

        return [position_t1,velocity_t1] , reward, done

    def render(self,file_path='./mountain_car.mp4',mode='mp4'):
        fig = plt.figure()
        ax = fig.add_subplot(111,autoscale_on=False, xlim=(-1.2,0.5),ylim=(-1.1,1.1))
        ax.grid(False)
        x_sin=np.linspace(start=-1.2,stop=0.5, num=100)
        y_sin= np.sin(3*x_sin)

        ax.plot(x_sin,y_sin)
        dot,=ax.plot([],[],'ro')
        time_text= ax.text(0.05,0.9,'',transform=ax.transAxes)
        _position_list=self.position_list
        _delta_t=self.delta_t

        def _init():
            dot.set_data([], [])
            time_text.set_text('')
            return dot, time_text

        def _animate(i):
                x = _position_list[i]
                y = np.sin(3 * x)

                dot.set_data([x], [y])
                time_text.set_text(
                    f"Time: {np.round(i * _delta_t, 1)}s\nFrame: {i}"
                )
                return dot, time_text

        if not self.position_list:
            raise ValueError(
                "Cannot render: no positions recorded. "
                "Call reset() and step() first."
            )

        ani = animation.FuncAnimation(
                fig,
                _animate,
                frames=range(len(self.position_list)),
                blit=True,
                init_func=_init,
                repeat=False,
                cache_frame_data=False
            )

        if mode == 'gif':
            ani.save(file_path, writer='pillow', fps=int(1 / self.delta_t))
        elif mode == 'mp4':
            ani.save(file_path, writer='ffmpeg', fps=int(1 / self.delta_t))
        else:
            raise ValueError("mode must be 'gif' or 'mp4'")
        fig.clear()
        plt.close(fig)
            
