
import matplotlib; #matplotlib.use("TkAgg")
import NSL as NSL
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

gaia = NSL.Satellite()
lineData = [gaia.z_]

days = 365*5
num_points = days * 100 + 1

j = 0

for i in np.arange(0,days,0.01):
    j = j+1
    gaia.Update(0.01)
    if j == 100:
        j = 0
        lineData.append(gaia.z_)
print("calculated zs")

num_points = len(lineData)

lineData = np.swapaxes(lineData,0,1)

def update_lines2(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(lineData[0:2, :num])
        line.set_3d_properties(lineData[2, :num])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Creating line object.
lines = [ax.plot(lineData[0, 0:1], lineData[1, 0:1], lineData[2, 0:1])[0]]


# Setting the axes properties
ax.set_xlim3d([-1.0, 1.0])
ax.set_xlabel('l')

ax.set_ylim3d([-1.0, 1.0])
ax.set_ylabel('m')

ax.set_zlim3d([-1.0, 1.0])
ax.set_zlabel('n')

ax.set_title('3D Plot of Evolving Z Axis')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines2, num_points, fargs=(lineData, lines),
                                   interval=1, blit=False, repeat=True)


#line_ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()