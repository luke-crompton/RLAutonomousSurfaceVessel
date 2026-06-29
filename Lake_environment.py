
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from Configuration import SimConfig, precompute_ray_lines

class LakeMapEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array'], 'render_fps': 20}


#cfg generates an instance of simconfig essentially making sure the class is always sim config
    def __init__(self, cfg: SimConfig, seed: int= 0, render_mode: str | None=None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.size = cfg.size

        self.obstacle = None
        self.visited_count = None
        self.disp_window = deque(maxlen=12)
        self.best_cov = 0
        self.steps_since_improve = 0
        self.progress_tolerance = 1e-3
        self.target_coverage = 0.7
        self.no_progress_lim = 1500
        self.best_steps = None
        #state
        self.x = self.y = self.heading = self.xf = self.yf = None
        self.agent_dt = 0.2
        self.physics_substeps = 4
        self.dt = self.agent_dt/ self.physics_substeps
        self.substep_telemetry = []

        #dynamics
        self.v = 0.0
        self.v_prev = 0.0
        self.yaw = 0.0 #rad/s
        self.turn_deadzone = 0.02

        self.max_accel = 0.35
        self.max_decel = 0.8
        self.k_drag = 0.15
        self.v_max = 1.0
        self.v_back = 0.2

        self.r_min = 2.5 #min turn radius at speed (in cells)
        self.yaw_cap= np.deg2rad(90) #hard yaw cap
        self.yaw_accel = np.deg2rad(80)# yaw rate accel rad/s^2
        
        self.applied_speed_cmd = 0.0       # m/s, filtered target speed sent to plant
        self.applied_yaw_cmd = 0.0         # rad/s, filtered target yaw-rate sent to plant

    # command slew limits per agent step (0.2 s)
        self.max_speed_cmd_step = 0.12     # m/s per agent step
        self.max_yaw_cmd_step = np.deg2rad(12.0)   # rad/s per agent step
        self.max_yaw_cmd_reverse_step = np.deg2rad(6.0)  # stricter when changing yaw sign

        self.beta_v_crash = 1 #scale crash reward with speed
        self.k_prox_v = 0.01 #proximity penalty scales with speed
        self.k_smooth = 0.01 #action smoothness penalty
        self.k_curv = 0.02 # penalising high curve at speed

        self.prev_throttle = 0
        self.prev_turn = 0

        #cone configuration
        self.N_cones = cfg.N_cones
        self.r_cone = cfg.r_cone
        #cone masks is N,S,S , each [i,:,:] in boolean for selecting cells that belong to cone i
        #in the circular patch around the model, cone counts is how many cells fall into each cone
        # its length N
        self.cone_masks, self.cone_counts = self.build_cone_masks(self.r_cone, self.N_cones)

        #buffers for information to go into cones
        #used for taking a copy of the local area so don't need to query the whole grid
        S = 2*self.r_cone +1
        self.buffer_free_cells = np.zeros((S,S), np.uint8)
        self.buffer_visited_cells = np.zeros((S,S), np.uint8)
        self.buffer_log_prob = np.zeros((S,S), np.float32)

        #known cells from what the model has "seen" of self.size by size
        self.known_free = None
        self.known_visited = None

        self.us_ray_max = self.cfg.us_ray_max
        self.us_ray_angle = self.cfg.us_ray_angle
        self.radar_angle = self.cfg.radar_angle
        self.radar_max = self.cfg.radar_max
        self.tof_max = self.cfg.TOF_max
        self.tof_angle = self.cfg.TOF_angle

        #precomputed dx,dx for all ray angles cached for cheap and quick instant access
        self.us_rays, self.us_offsets, self.us_theta_step, self.us_beam_degrees = precompute_ray_lines(
            self.us_ray_max,self.cfg.K_headings, float(self.us_ray_angle))

        self.k_headings = self.cfg.K_headings
        self.radar_rays, self.radar_offsets, self.radar_theta_step, self.radar_beam_degrees = precompute_ray_lines(
            self.radar_max,self.cfg.K_headings, float(self.radar_angle))

        self.tof_rays, self.tof_offsets, self.tof_theta_step, self.tof_beam_degrees = precompute_ray_lines(
            self.tof_max, self.k_headings, float(self.tof_angle))
        
        
        self.log_prob = np.zeros((self.size, self.size), dtype = np.float32)
        self.l_min = -4
        self.l_max = 4
        self.p_occ = self.cfg.p_occupied
        self.p_free = self.cfg.p_free
        self.l_occupied = np.log(self.p_occ/ (1-self.p_occ))
        self.l_free = np.log(self.p_free/ (1- self.p_free))
        
        self.log_occ_threshold = self.l_occupied
        
        xs, ys = np.mgrid[-self.r_cone:self.r_cone+1, -self.r_cone:self.r_cone+1]
        self.local_r = np.sqrt(xs**2 + ys**2).astype(np.float32)
        
        # obs: [xf,yf, sinh, cosh, tof_l, d_l, d_f, d_r, tof_r, v_norm, yaw_norm, action_speed, action_yaw] + N_cones*2
        obs_core = 13
        obs_cones = self.N_cones * 2
        obs_dimension = obs_core + obs_cones
        low = np.zeros(obs_dimension, np.float32)
        high = np.ones(obs_dimension, np.float32)

        for index in (2,3,9,10):
            low[index]= -1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

       # actions = [desired_speed, desired_yaw_rate]
        # both normalised to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.t=0

    def _gen_map(self):
        #creates an array size*size of number and places an object there
        # if it is less than the probability of an object_p in boolean terms
        self.obstacle = self.rng.random((self.size,self.size)) < self.cfg.obstacle_p

        # the same here but ~ (not operator) so only places where there is no obstacle
        self.algae = (self.rng.random((self.size,self.size)) < self.cfg.algae_p) & (~self.obstacle)
        # c is the center it clears a 5*5 grid so the boat always has a place to spawn
        c= self.size // 2; self.obstacle[c-2:c+3, c-2:c+3] = False

    def _spawn(self): # random location to spawn with a random heading, cant spawn on an obstacle
        #free is positions of all non obstacle coordinates ,
        # argwhere returns the row and collum of the true values
        free = np.argwhere(~self.obstacle)
        #one value that points to a row in the free array
        ix = self.rng.integers(0,len(free))
        #selects a row from the free array and applies the int function
        #numpy number aren't integers sometimes, map applies to function every number in the free[ix]
        self.x, self.y = map(int, free[ix])
        #giving the model a float x,y values so values don't have to map to whole numbers
        self.xf, self.yf = float(self.x), float(self.y)
        #returns a float between (a,b) so for this -180 and 180
        self.heading = float(self.rng.uniform(-np.pi, np.pi))

    def reset(self, *, seed= None , options = None):
        # the * makes sure that seed and option get called by name (error proofing)
        #generates a random seed if not selected
        if seed is not None:
            self.rng = np.random.default_rng(seed+ int(self.rng.integers(1e6)))
        else:
            self.rng = np.random.default_rng(np.random.randint(1e9))
        #generate map and spawn in
        self._gen_map(); self._spawn()
        #resets visited to array of false of size*size
        self.visited_count = np.zeros((self.size, self.size),dtype=np.int32)
        self.current_cell = None
        self.disp_window = deque(maxlen=12)
        self.coverage = 0.0
        self.best_cov =0
        self.steps_since_improve = 0
        self.substep_telemetry = []


        self.v = 0.0
        self.v_prev = 0.0
        self.yaw = 0.0
        self.prev_throttle = 0.0
        self.prev_turn = 0.0
        
        self.applied_speed_cmd = 0.0
        self.applied_yaw_cmd = 0.0
        

        self.known_free = np.zeros((self.size, self.size), dtype=np.uint8)
        self.known_visited = np.zeros((self.size, self.size), dtype=bool)
        self.log_prob = np.full((self.size, self.size),self.l_free*2 , dtype=np.float32)
        self.prev_coverage = float(self.known_visited.mean())

        self.t = 0
        #returns ...........
        return self._obs(), {"covered": 0.0}

    def index_from_heading(self,theta_step, heading):
        bin_num = int(np.round((heading % (2*np.pi)) / theta_step)) # picks the nearest k bin
        valid_bin_num = bin_num % self.k_headings # bin number wrapped to k heading to keep in valid range
        return valid_bin_num


    def accurate_cast_ray_line(self, angle, ray_max, collect_cells= False):
        x, y = float(self.x), float(self.y)
        dx, dy = np.cos(angle), np.sin(angle)

        if abs(dx)< 1e-6 and abs(dy)< 1e-6:
            return float(ray_max), (set() if collect_cells else None)

        ix, iy = int(np.floor(x)), int(np.floor(y))
        #determines what direction to step x and y in ie if dy is < 0 then its stepping in -1
        stepx =1 if dx > 0 else -1 if dx < 0 else 0
        stepy =1 if dy > 0 else -1 if dy < 0 else 0



        if stepx != 0:
            next_hline = (ix+ (stepx> 0 ))-x
            time_max_x = next_hline/ dx
            time_delta_x = stepx/ dx
        else:
            time_max_x = float('inf')
            time_delta_x = float('inf')

        if stepy != 0:
            next_vline = (iy+ (stepy> 0 ))-y
            time_max_y = next_vline/ dy
            time_delta_y = stepy/ dy
        else:
            time_max_y = float('inf')
            time_delta_y = float('inf')

        t =0.0
        seen = set() if collect_cells else None

        for _ in range(int(ray_max) +2):
            if time_max_x < time_max_y:
                t = time_max_x; time_max_x += time_delta_x; ix += stepx
            else:
                t = time_max_y; time_max_y += time_delta_y; iy += stepy

            if t >= ray_max:
                return (float(ray_max), seen) if collect_cells else float(ray_max)


            if ix < 0 or iy < 0 or ix >= self.size or iy >= self.size:
                return (min(float(ray_max), max(0.0, t)), seen) if collect_cells else min(float(ray_max), max(0.0, t))

            if collect_cells:
                seen.add((ix,iy))

            if self.obstacle[ix, iy]:
                return (min(float(ray_max), max(0.0, t)), seen) if collect_cells else min(float(ray_max), max(0.0, t))


        return (float(ray_max),seen) if collect_cells else float(ray_max)

    def _new_ray_fan(self, heading, rays, offsets, theta_step, ray_max,
                         count_unvisited: bool = False, write_known_free: bool = False):
        if rays is self.us_rays and offsets is self.us_offsets and theta_step == self.us_theta_step:
            beam_degrees = self.us_beam_degrees
        elif rays is self.radar_rays and offsets is self.radar_offsets and theta_step == self.radar_theta_step:
            beam_degrees = self.radar_beam_degrees
        else:
            beam_degrees = self.tof_beam_degrees

        k = self.index_from_heading(theta_step, heading)
        base = k * theta_step
        
        beam_radians = np.deg2rad(beam_degrees)
        n_beams = beam_radians.size


        min_d = ray_max

        if not count_unvisited:
            # offset[k] is start , end indexes
            for beam in range(n_beams):
                angle = base + beam_radians[beam]
                distance = self.accurate_cast_ray_line(angle, ray_max, collect_cells=False)
                if distance < min_d:
                    min_d = distance
                    if min_d <= 1e-6:
                        break
            return float(min_d)


        # ensuring im not writing distances beyond what the ray can infer
        dists, seen_list, angle_list = [],[], []
        for beam in range(n_beams):
            angle = base + beam_radians[beam]
            distance, seen = self.accurate_cast_ray_line(angle, ray_max, collect_cells=True)
            dists.append(distance)
            seen_list.append(seen if seen is not None else set())
            angle_list.append(angle)
            if distance < min_d:
                min_d = distance
                
                
        free_updates = set()
        hit_updates= set()
                
        for dist, cells, angle in zip(dists, seen_list, angle_list):
            
            cell_list = list(cells)
            if dist < ray_max:
                free_cells = cell_list[:-1]
            else:
                free_cells = cell_list
                
            free_updates.update(free_cells)
            
                
            if dist < ray_max and cells:
                
                hx = self.xf + np.cos(angle)* dist
                hy = self.yf + np.sin(angle) * dist
                hx_int = int(np.floor(hx))
                hy_int = int(np.floor(hy))
                
                if 0<= hx_int< self.size and 0<= hy_int <self.size:
                    hit_updates.add((hx_int,hy_int))
                    
        for (ix,iy) in free_updates - hit_updates:
            self.update_object_p(ix, iy, hit = False)
            
        for (ix,iy) in hit_updates:
            self.update_object_p(ix, iy, hit= True)

        eps = 1e-9
        all_seen = set()
        for distance, cells in zip(dists, seen_list):
            if distance <= (min_d +eps):
                if distance< ray_max and cells:
                    cell_list = list(cells)
                    all_seen.update(cell_list[:-1])
                else:
                    all_seen.update(cells)
        visible_cells = len(all_seen)
        if visible_cells ==0:
            fraction = 0.0
        else:
            unvisited = sum(1 for (ix,iy) in all_seen if not self.known_visited[ix,iy])
            fraction = float(unvisited) / float(visible_cells)

        if write_known_free and visible_cells > 0:
            for (ix,iy) in all_seen:
                if self.log_prob[ix,iy] < self.log_occ_threshold:
                    self.known_free[ix,iy] = True

        return float(min_d), float(np.clip(fraction,0,1)), visible_cells


    def us_ray_fan(self, heading, count_unvisited = True, write_known_free = True):
        return self._new_ray_fan(heading,self.us_rays, self.us_offsets, self.us_theta_step,
                                 self.us_ray_max,count_unvisited, write_known_free)

    def radar_ray_fan(self, heading, count_unvisited = True, write_known_free = True):
        return self._new_ray_fan(heading,self.radar_rays, self.radar_offsets, self.radar_theta_step,
                                 self.radar_max,count_unvisited, write_known_free)
    
    def tof_ray_fan(self, heading, count_unvisited= True, write_known_free= True):
        return self._new_ray_fan(heading,self.tof_rays, self.tof_offsets, self.tof_theta_step,
                                 self.tof_max, count_unvisited=True, write_known_free= True)


    def _translate_actions(self, action):
        a = np.array(action, dtype=float).flatten()
        speed_cmd = float(np.clip(a[0], -1.0, 1.0))
        yaw_cmd = float(np.clip(a[1], -1.0, 1.0))
    
        # map normalised command to physical target speed
        # forward uses v_max, reverse uses v_back
        if speed_cmd >= 0.0:
            desired_speed = speed_cmd * self.v_max
        else:
            desired_speed = speed_cmd * self.v_back
    
        # map normalised yaw command to desired yaw rate
        desired_yaw_rate = yaw_cmd * self.yaw_cap
    
        # optional deadzone on yaw-rate command
        if abs(desired_yaw_rate) < self.turn_deadzone * self.yaw_cap:
            desired_yaw_rate = 0.0
    
        return desired_speed, desired_yaw_rate
    
    def _slew_limit(self, target, current, max_step):
        delta = target - current
        delta = np.clip(delta, -max_step, max_step)
        return current + delta
    
    
    def _apply_command_limits(self, desired_speed, desired_yaw_rate):
        # limit desired speed change
        self.applied_speed_cmd = self._slew_limit(
           desired_speed,
           self.applied_speed_cmd,
           self.max_speed_cmd_step
           )

        # use stricter limit if yaw command changes sign
        yaw_sign_change = (
            abs(self.applied_yaw_cmd) > 1e-6 and
            abs(desired_yaw_rate) > 1e-6 and
            np.sign(desired_yaw_rate) != np.sign(self.applied_yaw_cmd)
        )

        yaw_step_limit = (
            self.max_yaw_cmd_reverse_step if yaw_sign_change
            else self.max_yaw_cmd_step
            )

        self.applied_yaw_cmd = self._slew_limit(
            desired_yaw_rate,
            self.applied_yaw_cmd,
            yaw_step_limit
            )

        return self.applied_speed_cmd, self.applied_yaw_cmd

    def _update_speed(self, desired_speed):
        speed_error = desired_speed - self.v
    
        if speed_error >= 0.0:
            dv = min(speed_error, self.max_accel * self.dt)
        else:
            dv = max(speed_error, -self.max_decel * self.dt)
    
        # optional quadratic drag
        drag = self.k_drag * self.v * abs(self.v) * self.dt
    
        self.v_prev = self.v
        self.v = self.v + dv
    
        # apply drag after command response
        if self.v > 0:
            self.v = max(0.0, self.v - drag)
        elif self.v < 0:
            self.v = min(0.0, self.v - drag)
    
        self.v = np.clip(self.v, -self.v_back, self.v_max)

    def _yaw_update(self, desired_yaw_rate):
        # global cap
        desired_yaw_rate = np.clip(desired_yaw_rate, -self.yaw_cap, self.yaw_cap)
    

        yaw_by_r = abs(self.v) / max(1e-6, self.r_min)
        yaw_limit = min(self.yaw_cap, max(np.deg2rad(20.0), yaw_by_r))
    
        desired_yaw_rate = np.clip(desired_yaw_rate, -yaw_limit, yaw_limit)
    
        yaw_error = desired_yaw_rate - self.yaw
        dyaw = np.clip(yaw_error, -self.yaw_accel * self.dt, self.yaw_accel * self.dt)
    
        self.yaw = np.clip(self.yaw + dyaw, -self.yaw_cap, self.yaw_cap)
        
    def _proximity_penalty(self, min_d):
        if min_d >= 0.6: return 0.0
        scale = np.clip((min_d -0.2)/0.5, 0, 1)
        #scale 0.25 + 0.75 if the boats v is small it still get penalised
        speed_scale = 0.25 + 0.75 * abs(self.v/ max(1e-6, self.v_max))
        return -self.k_prox_v * (1.0- scale) * speed_scale

    def _smooth_penalty(self, speed_cmd, yaw_cmd):
        delta_speed = speed_cmd - self.prev_throttle
        delta_yaw   = yaw_cmd   - self.prev_turn

        penalty = -(
            0.0 * (delta_speed ** 2) +
            0.0 * (delta_yaw ** 2))

        self.prev_throttle = speed_cmd
        self.prev_turn = yaw_cmd
        return penalty

    def _speed_turning_penalty(self):
        return - self.k_curv * abs(self.v) * abs(self.yaw)

    def build_cone_masks(self, R , N):
        #very confusing at first glance so heavily commented
        #setting every x and y coordinate in that 2R+1 patch between -pi and pi
        xs , ys = np.mgrid[-R:R+1,-R:R+1]
        r_squared = (xs**2) + (ys**2)
        #boolean mask for whether a point is actually inside the circle radius
        inside = r_squared <= (R**2)
        # angle gives the polar angle of every grid, mapping between 0,2pi
        angle = (np.arctan2(ys,xs ) + 2*np.pi) %(2*np.pi)
        #converting angle into int bins from 0,(N-1), tells you which sector that grid is in
        bins = (angle / (2*np.pi)* N).astype(np.int32) % N # Mod N to round numbers
        #N is cone angle index , x coord in patch and y coord in patch (2R + 1 for full diameter)
        masks = np.zeros((N,(2*R)+1,(2*R)+1), dtype= bool)

        for cone in range(N):#adding each grid to each cone index for each angle in the cone
            masks[cone] = inside & (bins == cone)

        masks[:, R,R]= False # centre has no angle(=0) so undefined
        #number of grids in each cone (N,-1) lets numpy infer the dimension automatically
        #sum axis =1 adds up the number of grids in cone N,
        counts = masks.reshape(N, -1).sum(axis=1).astype(np.float32).clip(1)
        return masks, counts

    def local_extract(self, Map, cx, cy, R, out_buffer):
        #inputs: a global map M, centre cx,cy and radius
        #outputs a buffer of size (2r +1)^2 and copies the local region wanted from map M
        #had to make this as to not loop through entire map index everytime I want to update the cones
        #so this function returns the coordinates needed to query local maps
        height, width = self.size, self.size
        out_buffer.fill(0)

        #max and min x and y coordinates relative to current position
        x0 ,x1 = cx-R , cx+(R+1)
        y0 , y1 = cy-R , cy+(R+1)

        #coordinates checking not oob
        sx0 = max(0, x0); sx1 = min(width, x1)
        sy0 = max(0, y0); sy1 = min(height, y1)

        #target coordinates
        #example if cx=3 and R=5 then x0 is -2 so tx0 is +2,
        # so +2 is the index of out buffer where the data should start from
        #as outbuffer is of size 2R+1 by 2R+1 but the valid data might be less if model is near an edge
        tx0 = sx0 - x0; ty0 = sy0 - y0

        tx1 = tx0 +(sx1- sx0); ty1 = ty0 + (sy1- sy0)

        if sx1> sx0 and sy1> sy0: # only update buffer if there is any valid points at all (safety net)
            out_buffer[tx0:tx1, ty0:ty1] = Map[sx0:sx1, sy0:sy1]
        return out_buffer

    def compute_cones(self):
        #returns free fraction , visited fraction, algae seen fraction

        cx , cy = int(round(self.xf)), int(round(self.yf))
        radius = self.r_cone
        num_cones = self.N_cones

        #pulling known free cells, known visited cells and cells with known algae
        #while only querying from global map M at the local region I want to look at
        self.local_extract(self.known_free, cx, cy, radius, self.buffer_free_cells)
        self.local_extract(self.known_visited, cx, cy, radius, self.buffer_visited_cells)

        
        self.local_extract(self.log_prob, cx, cy, radius, self.buffer_log_prob)
        
        
        
        unvisited_cells  = (self.buffer_free_cells ==1) & (self.buffer_visited_cells == 0)

        sum_unvisited = (self.cone_masks* unvisited_cells).reshape(num_cones,-1).sum(axis=1)
        
        counts = self.cone_counts
        
        unvisited_frac = (sum_unvisited/ counts).clip(0,1)
        
        
        r = self.local_r
        
        cone_distance_norm = np.empty(num_cones, dtype= np.float32)
        
        for k in range(num_cones):
            mask = self.cone_masks[k] & (self.buffer_log_prob >= self.log_occ_threshold)
            
            if np.any(mask):
                min_r = r[mask].min()
                
            else:
                min_r = float(radius)
                
            cone_distance_norm[k]= np.clip(min_r/ float(radius), 0.0, 1.0)
            
        
                
        
        
        return unvisited_frac, cone_distance_norm

    def translate_egocentric_cone(self, cone_vector):
        #split circle into N equal bins
        cone_step = (2*np.pi) / self.N_cones
        #wrap current heading to 0,2pi
        # over cone_step converts heading angle into a cone index
        #round to nearest index and % N cones in case rounding produced N cones
        rotate = int(np.round((self.heading%(2*np.pi))/ cone_step)) % self.N_cones
        #np.roll rotates the cone values in a circle so the cone that points closest to my heading is at index 0
        return np.roll(cone_vector, -rotate)

    def sensor_noise(self, norm_dist, sigma=0.02, p_dropout=0.02, q_step=0.02, bias=0.0):
        if np.random.random() < p_dropout:
            return 1.0
        #adds random noise from (0, sigma) then between 0,1
        norm_dist = np.clip(norm_dist +bias + np.random.normal(0,sigma), 0, 1)

        if q_step is not None and q_step > 0:
            #q step is sensor resolution so the results don't become floating point numbers
            bins = np.round(norm_dist / q_step)
            norm_dist = np.clip(bins* q_step, 0, 1)
        return norm_dist


    def update_object_p(self, ix,iy, hit =  False):
      
        #every step add or take away a probability an object is there

        if hit:
            self.log_prob[ix, iy] = np.clip(self.log_prob[ix, iy] + self.l_occupied,
                                        self.l_min, self.l_max)
        else:
            self.log_prob[ix, iy] = np.clip(self.log_prob[ix, iy] + self.l_free,
                                        self.l_min, self.l_max)





    def _obs(self):#calling all observation functions and setting visited to true for current pos
        self.known_visited[self.x, self.y] = True
        d_f, _, _ = self.radar_ray_fan(self.heading)
        d_l, _, _ = self.us_ray_fan(self.heading - np.pi/6)
        d_r, _, _ = self.us_ray_fan(self.heading + np.pi/6)

        d_f = d_f / self.radar_max
        d_l = d_l / self.us_ray_max
        d_r = d_r / self.us_ray_max

        d_f = self.sensor_noise(d_f, q_step = None)
        d_l = self.sensor_noise(d_l, q_step = None)
        d_r = self.sensor_noise(d_r, q_step = None)

        tof_l, _, _ = self.tof_ray_fan(self.heading - (np.pi/2))
        tof_r,_, _ = self.tof_ray_fan(self.heading + (np.pi/2))
        
        tof_l = tof_l / self.tof_max
        tof_r = tof_r / self.tof_max

        v_norm = np.clip(self.v/self.v_max, -1.0, 1.0)
        yaw_norm = np.clip(self.yaw/ self.yaw_cap, -1.0 , 1.0)
        
        prev_action_speed = self.prev_throttle
        prev_action_yaw = self.prev_turn

        core =np.array([
            self.xf/(self.size-1), self.yf/(self.size-1),
            np.sin(self.heading), np.cos(self.heading),
            tof_l, d_l, d_f, d_r, tof_r , v_norm, yaw_norm, prev_action_speed, prev_action_yaw],
            np.float32)


        cone_free, cone_ob_d = self.compute_cones()

        cone_ob_d = self.translate_egocentric_cone(cone_ob_d)
        cone_free = self.translate_egocentric_cone(cone_free)
        
        

        return np.concatenate([core , cone_free, cone_ob_d]).astype(np.float32)

    def step(self, action):
        self.t += 1
        obs_now = self._obs()
        d_l, d_f, d_r = obs_now[5], obs_now[6], obs_now[7]
        min_d = min(abs(d_f), abs(d_l), abs(d_r))

        a = np.array(action, dtype=float).flatten()
        speed_cmd = float(np.clip(a[0], -1.0, 1.0))
        yaw_cmd   = float(np.clip(a[1], -1.0, 1.0))

        r_smoothing = self._smooth_penalty(speed_cmd, yaw_cmd)
        desired_speed, desired_yaw_rate = self._translate_actions(action)
        desired_speed, desired_yaw_rate = self._apply_command_limits(desired_speed, desired_yaw_rate)
        
        terminated = False
        truncated = False

        r_covered = 0.0
        r_stall = 0.0
        r_crash = 0.0
        r_finish = 0.0

        total_disp = 0.0
        crashed = False
        ix = self.x
        iy = self.y

        self.substep_telemetry = []
        for _ in range(self.physics_substeps):
            self._update_speed(desired_speed)
            self._yaw_update(desired_yaw_rate)

            heading_next = (self.heading + self.yaw * self.dt) % (2 * np.pi)
            nfx = self.xf + np.cos(heading_next) * self.v * self.dt
            nfy = self.yf + np.sin(heading_next) * self.v * self.dt

            sub_disp = np.hypot(nfx - self.xf, nfy - self.yf)
            total_disp += sub_disp

            ix = int(np.floor(nfx))
            iy = int(np.floor(nfy))

            crashed = (
                    ix < 0 or iy < 0 or ix >= self.size or iy >= self.size
                    or self.obstacle[ix, iy]
            )

            if crashed:
                speed_factor = 1.0 + self.beta_v_crash * max(0.0, abs(self.v) / max(1e-6, self.v_max))
                r_crash = -5.0 * speed_factor
                terminated = True
                break

            self.xf, self.yf = nfx, nfy
            self.x, self.y = ix, iy
            self.heading = heading_next

            self.substep_telemetry.append({
                "v": float(self.v),
                "yaw": float(self.yaw),
                "xf": float(self.xf),
                "yf": float(self.yf),
                "heading": float(self.heading),
                "crashed": bool(crashed),
            })

            for x in range(-1, 2):
                for y in range(-1, 2):
                    nx, ny = ix + x, iy + y
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if not self.obstacle[nx, ny] and not self.known_visited[nx, ny]:
                            self.known_visited[nx, ny] = True

        self.disp_window.append(total_disp)

        if not crashed:
            enter_new_cell = (self.current_cell != (ix, iy))
            if enter_new_cell:
                self.current_cell = (ix, iy)
            else:
                if (
                        len(self.disp_window) == self.disp_window.maxlen
                        and sum(self.disp_window) < 0.6
                        and abs(self.yaw) < np.deg2rad(4)
                ):
                    r_stall = -0.02

        r_proximity = self._proximity_penalty(min_d)

        if np.sign(self.v) != np.sign(self.v_prev):
            r_reverse = -0.007
        else:
            r_reverse = 0.0

        coverage_now = float(self.known_visited.mean())
        coverage_gain = max(0.0, coverage_now - self.prev_coverage)
        r_covered = 500.0 * coverage_gain
        self.prev_coverage = coverage_now

        if coverage_gain <= 0:
            reward = -0.0025
        else:
            reward = 0.0

        self.coverage = coverage_now
        if self.coverage > self.best_cov + self.progress_tolerance:
            self.best_cov = self.coverage
            self.steps_since_improve = 0
        else:
            self.steps_since_improve += 1

        if not crashed:
            if self.coverage >= self.target_coverage:
                r_finish = 5.0
                speed_scale = 5.0
                if self.best_steps is None:
                    self.best_steps = self.t
                    speed_bonus = 0.0
                else:
                    improvement = max(0.0, (self.best_steps - self.t) / self.best_steps)
                    speed_bonus = speed_scale * improvement
                    if self.t < self.best_steps:
                        self.best_steps = self.t

                r_finish += speed_bonus
                truncated = True

            elif self.steps_since_improve >= self.no_progress_lim:
                truncated = True
                r_stall += -1.0

            elif self.t >= self.cfg.max_steps:
                truncated = True

        reward += r_covered
        reward += r_crash
        reward += r_stall
        reward += r_proximity
        reward += r_smoothing
        reward += r_reverse
        reward += r_finish

        obs = self._obs()

        info = {
            "coverage": float(self.known_visited.mean()),
            "ended_by": (
                "crash" if crashed else
                ("coverage" if self.known_visited.mean() >= 0.95 else
                 ("timeout" if self.t >= self.cfg.max_steps else "running"))
            ),
            "r_covered": r_covered,
            "r_crash": r_crash,
            "r_stall": r_stall,
            "r_proximity": r_proximity,
            "r_smoothing": r_smoothing,
            "r_reverse": r_reverse,
            "r_finish": r_finish,
            "total_reward": reward,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        #making array of current state of the grid
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        img[self.known_visited] = [40,40,40]
        img[self.algae] = [0,200,0]
        img[self.obstacle] = [200,0,0]
        return img

