
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import numpy as np
from gym import spaces


class Drivingbehavior:

    def __init__(self, env, trafficstate, min_SDHE, min_SDspeed, min_meanHE, max_SDLP, begin_time, sumo):
        self.trafficstate = trafficstate
        self.env = env
        self.min_SDHE = min_SDHE
        self.min_SDspeed = min_SDspeed
        self.min_meanHE = min_meanHE
        self.max_SDLP = max_SDLP
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_reward = None
        self.sumo = sumo

        self.build_phases()

        self.lanes = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.trafficstate)))  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.trafficstate) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes}

        self.observation_space = spaces.Box(low=np.zeros(self.traffic_phases + 1 + 2 * len(self.lanes), dtype=np.float32), high=np.ones(self.traffic_phases + 1 + 2 * len(self.lanes), dtype=np.float32))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.traffic_phases),
            spaces.Discrete(2),                                           # Binary variable active if min_meanHE seconds already elapsed
            *(spaces.Discrete(10) for _ in range(2*len(self.lanes)))      # Density and stopped-density for each lane
        ))
        self.action_space = spaces.Discrete(self.traffic_phases)

    def build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.trafficstate)[0].phases
        if self.env.fixed_ts:
            self.traffic_phases = len(phases) // 2
            return

        self.speed_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if 'y' not in state and (state.count('r') + state.count('s') != len(state)):
                self.speed_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.traffic_phases = len(self.speed_phases)
        self.all_phases = self.speed_phases.copy()

        for i, p1 in enumerate(self.speed_phases):
            for j, p2 in enumerate(self.speed_phases):
                if i == j: continue
                yellow_state = ''
                for s in range(len(p1.state)):
                    if (p1.state[s] == 'G' or p1.state[s] == 'g') and (p2.state[s] == 'r' or p2.state[s] == 's'):
                        yellow_state += 'y'
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i,j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.min_SDspeed, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.trafficstate)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.trafficstate, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.trafficstate, self.all_phases[0].state)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step
    
    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.min_SDspeed:
            #self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.trafficstate, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, new_phase):

        new_phase = int(new_phase)
        if  self.green_phase == new_phase or self.time_since_last_phase_change < self.min_SDspeed + self.min_meanHE:
            self.sumo.trafficlight.setRedYellowGreenState(self.trafficstate, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.min_SDHE
        else:

            self.sumo.trafficlight.setRedYellowGreenState(self.trafficstate, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state)
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.min_SDHE
            self.is_yellow = True
            self.time_since_last_phase_change = 0
    
    def compute_observation(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.traffic_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_meanHE + self.min_SDspeed else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation
            
    def compute_reward(self):
        self.last_reward = self._waiting_time_reward()
        return self.last_reward
    
    def _pressure_reward(self):
        return -self.get_pressure()

    def _queue_average_reward(self):
        new_average = np.mean(self.get_stopped_vehicles_num())
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def _queue_reward(self):
        return - (sum(self.get_stopped_vehicles_num()))**2

    def _waiting_time_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _waiting_time_reward2(self):
        ts_wait = sum(self.get_waiting_time())
        self.last_measure = ts_wait
        if ts_wait == 0:
            reward = 1.0
        else:
            reward = 1.0/ts_wait
        return reward

    def _waiting_time_reward3(self):
        ts_wait = sum(self.get_waiting_time())
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward

    def get_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_pressure(self):
        return abs(sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes) - sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes))

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepVehicleNumber(lane) / (self.sumo.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.out_lanes]

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepVehicleNumber(lane) / (self.lanes_length[lane] / vehicle_size_min_gap)) for lane in self.lanes]

    def get_lanes_queue(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepHaltingNumber(lane) / (self.lanes_length[lane] / vehicle_size_min_gap)) for lane in self.lanes]
    
    def get_total_queued(self):
        return sum([self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list
