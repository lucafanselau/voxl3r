import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from datasets.chunk.image_heuristics import AngleHeuristics

# Create an instance of AngleHeuristics
angle_heuristics = AngleHeuristics()

# Generate angles from 0 to pi
angles = np.linspace(0, np.pi, 500)
cos_angles = np.cos(angles)

# Calculate heuristic values
values = [angle_heuristics.angle_to_value(cos_angle) for cos_angle in cos_angles]

# Calculate figure size with aspect ratio 10:3
fig_width, fig_height = figaspect(3/10)

# Plot the values
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.plot(angles, values, color='blue')

# Remove axis labels and title
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('')

# Remove background
ax.set_facecolor('none')
fig.patch.set_facecolor('none')

# Save the plot as a PNG file with a transparent background
plt.savefig('angle_heuristic_plot.png', transparent=True, bbox_inches='tight')
plt.close() 