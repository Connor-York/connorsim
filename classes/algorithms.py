import numpy as np 




def bresenhams_line(position, goal):
    x0, y0 = position
    x1, y1 = goal
    
    #gradients, slopes from top left to bottom right by default 
    dx = abs(x1 - x0) 
    dy = -abs(y1 - y0)
    
    num_steps = max([dx, -dy]) + 1

    coords = np.zeros(num_steps)
    
    sx = 1 if x0 < x1 else -1 #if decr in x axis, decr instead of incr
    sy = 1 if y0 < y1 else -1 #if decr in y axis, decr instead of incr
    
    error = dx + dy
    
    coords = np.zeros((2,num_steps), dtype=int) # np.array, two rows, x = coords[0][:] y = coords[1][:]
    
    for i in range(num_steps):
        coords[0][i] = x0
        coords[1][i] = y0
        
        e2 = 2*error
        if e2 >= dy: # if error in the y axis, x pixel needs to increment
            error += dy
            x0 += sx
        
        if e2 <= dx: # if error in the x axis, y pixel needs to increment 
            error += dx
            y0 += sy
    
    return coords
        
def slice_from_coords(world, coords):
    
    cols, rows = coords[0], coords[1]
    view = world[rows[1:], cols[1:]] # if you dont care about the start value (you should be there)
    view = view.flatten()
    
    return view

def ordinal_walls(world, position, new_coord):
    """
    takes a slice between position and new coord (ordinal directions only) and checks for walls within that slice
    returns: 
        False if a wall immediately infront, 
        new_coord if there are no walls, 
        or the coordinate just before the first wall
    """
        
    x1, y1 = position
    x2, y2 = new_coord
        
    #directions of view
    xstep = np.sign(x2-x1)
    ystep = np.sign(y2-y1)
    
    #view distances, manhattan
    ylength = abs(y2-y1) + 1
    xlength = abs(x2-x1) + 1
    
    #get cells coords to view
    rows = np.arange(y1 + ystep, y1 + (ylength * ystep), ystep) if ystep != 0 else np.array([y1])
    cols = np.arange(x1 + xstep, x1 + (xlength * xstep), xstep) if xstep != 0 else np.array([x1])
    
    view = world[rows, cols]
    view = view.flatten()
    
    walls = np.where(view==0)[0]
    
    if len(walls) == 0:
        return new_coord #do the full step
    elif walls[0] == 0:
        return False #theres a wall infront of you, don't step forward
    else:
        x_pos = x1 + xstep + (xstep * (walls[0] - 1))
        y_pos = y1 + ystep + (ystep * (walls[0] - 1))
        new_coord = (x_pos, y_pos)
        return new_coord # step just prior to the first wall 
    