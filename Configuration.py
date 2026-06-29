from dataclasses import dataclass
import numpy as np
import random
import os
from functools import lru_cache

#auto adds things to a plain class such as __init
@dataclass
class SimConfig:
    size: int = 64
    obstacle_p: float = 0.001
    algae_p: float= 0.03
    max_steps: int = 2000
    cell_size: float = 1
    look_ahead: int = 3
    us_ray_angle: float= 30
    us_ray_max: int = 4
    radar_angle: float  =30
    radar_max: int = 4
    UV_ray_max: int = 4
    TOF_max: int = 4
    TOF_angle: int = 30
    set_dt: float = 0.2
    drag_k: float = 0
    thrust_k: float = 1
    N_cones: int = 12
    r_cone: int = 4
    K_headings: int = 360
    p_occupied: float = 0.6
    p_free: float = 0.5

#setting a seed allows you to start from
# the same initial random state for reproducibility and debugging
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
#eg set_seed(10) will always be the same map


@lru_cache(maxsize=None)
def Precompute_line_step(maxr, angle):
    x, y = np.cos(angle), np.sin(angle)
    points = []
    last = None
    for r in range(1,maxr+1):
        dx = int(round(x*r))
        dy = int(round(y*r))
        if (dx, dy) != last:
            points.append((dx, dy))
            last = dx,dy
    return points

def precompute_ray_lines(maxr, k_headings, ray_angle):
    ray_ang = ray_angle

    half = ray_ang/2
    step = half/2
    #splitting ray angle into rays with seperation 1 (=1e-6 for rounding errors)
    beam_degrees = np.arange(-half,half + 1e-6, step , dtype=np.float64)
    #converting to radians
    beam_radians = np.deg2rad(beam_degrees)
    n_beams = beam_radians.size
    #what theta is stepped by
    theta_step = 2*np.pi/k_headings
    starts =[]
    dxs, dys = [], []

    #for all 360 degree lines at 1 degree intervals
    for k in range(k_headings):
        #starting K values goes up every loop
        base = k*theta_step
        for step in beam_radians:
            # for the rays centered at heading K will have different rays for that exact heading
            #this does include redundancy, but it's cheap to precompute and i may use this data later
            #to introduce lidar if that is something the group wants to do
            angle =  base +step
            points = Precompute_line_step(maxr, angle)#using precompute angles and returning (dx,dy) tuple
            start = len(dxs) # starting index of this new list of steps
            if points: # if points exists
                px, py = zip(*points) # zip returns (x0,y0), (x1,y1) so x = x0,x1 etc
                dxs.extend(px)# extend as adding a list to a list
                dys.extend(py)
            end = len(dxs)# ending index of this list
            starts.append((start,end))#appending index of starts

    #every dx , dy into one list so rays[i] = [dx_i, dy_i]
    rays = np.column_stack([ np.asarray(dxs, np.int16),
                             np.asarray(dys, np.int16)])
    #where offsets hold the index of start, end for offsets[k_heading, Beam]
    offsets = np.asarray(starts, np.int32).reshape(k_headings, n_beams, 2)

    #it is extremely quick and less memory intensive to just index a list with all precomputed values
    #this will speed up mapping and make it quicker to run many training loops
    return rays, offsets, theta_step, beam_degrees