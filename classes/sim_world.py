import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class World:
    """
    :class: 'World' represents the environment for the agents to explore. It includes methods to generate the gridworld, either bare or from a provided pgm
    """
    
    def __init__(self, map_name=None):
        
        self.gridworld = generate_world(map_name)
 
        # plt.imshow(self.gridworld, cmap='gray', vmin=0.0, vmax=1.0)
        # plt.show()
        
def generate_world(map_name):
    if map_name is not None:

        filename = f"maps/{map_name}/{map_name}.pgm"

        threshold = 0.95
        img = Image.open(filename)

        flipped_pixel_data = np.flipud(img)
        # Convert the image to a NumPy array
        img_np = np.asmatrix(flipped_pixel_data)

        # Normalize pixel values to range [0, 1]
        img_np = img_np / 255.0

        # Apply thresholding
        img_np[img_np >= threshold] = 1
        img_np[img_np < threshold] = 0

        return img_np
    
    else:
        
        # generate an empty world with black border
        
        grid = np.ones((600,600))
        grid = np.pad(grid, 10, 'constant', constant_values=0)
        
        return grid
        
        