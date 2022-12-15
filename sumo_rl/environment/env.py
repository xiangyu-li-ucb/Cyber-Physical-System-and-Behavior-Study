import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple
import sumo_rl
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
import gym
from gym.envs.registration import EnvSpec
import numpy as np
import pandas as pd

from .driving_behavior import Drivingbehavior

from gym.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ


def env(**kwargs):
    env = SumoEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)

path="../sumo_rl/environment/drivingdata/"
files=os.listdir(path)
print(files)

for file in files:
    position = path+file
    with open(position, "r") as f:  # 打开文件
        data = f.read()

class SumoEnvironment(gym.Env):

    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
        self,
        net_file: str,
        route_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        virtual_display: Optional[Tuple[int,int]] = None,
        begin_time: int = 0,
        num_seconds: int = 20000,
        max_depart_delay: int = 100000,
        time_to_teleport: int = -1,
        min_SDHE: int = 3,
        min_SDspeed: int = 0,
        min_meanHE: int = 5,
        max_SDLP: int = 50,
        single_agent: bool = False,
        sumo_seed: Union[str,int] = 'random',
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
    ):
        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self.virtual_display = virtual_display

        assert min_SDHE > min_SDspeed, "min_SDHE must be at least greater than min_SDspeed."

        self.begin_time = begin_time
        self.sim_max_time = num_seconds
        self.min_SDHE = min_SDHE  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.min_meanHE = min_meanHE
        self.max_SDLP = max_SDLP
        self.min_SDspeed = min_SDspeed
        self.single_agent = single_agent
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None

        if LIBSUMO:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net], label='init_connection'+self.label)
            conn = traci.getConnection('init_connection'+self.label)
        self.trafficstate = list(conn.trafficlight.getIDList())
        self.Drivingbehavior = {ts: Drivingbehavior(self, ts, self.min_SDHE, self.min_SDspeed, self.min_meanHE,
                                                    self.max_SDLP, self.begin_time, conn) for ts in self.trafficstate}
        conn.close()

        self.vehicles = dict()
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = EnvSpec('SUMORL-v0')
        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.trafficstate}
        self.rewards = {ts: None for ts in self.trafficstate}
    
    def _start_simulation(self):
        sumo_cmd = [self._sumo_binary,
                     '-n', self._net,
                     '-r', self._route,
                     '--max-depart-delay', str(self.max_depart_delay), 
                     '--waiting-time-memory', '10000',
                     '--time-to-teleport', str(self.time_to_teleport)]
        if self.begin_time > 0:
            sumo_cmd.append('-b {}'.format(self.begin_time))
        if self.sumo_seed == 'random':
            sumo_cmd.append('--random')
        else:
            sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append('--no-warnings')
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
            if self.virtual_display is not None:
                sumo_cmd.extend(['--window-size', f'{self.virtual_display[0]},{self.virtual_display[1]}'])
                from pyvirtualdisplay.smartdisplay import SmartDisplay
                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)
        
        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")                

    def reset(self):
        if self.run != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        self._start_simulation()

        self.Drivingbehavior = {ts: Drivingbehavior(self, ts, self.min_SDHE, self.min_SDspeed, self.min_meanHE,
                                                    self.max_SDLP, self.begin_time, self.sumo) for ts in self.trafficstate}
        self.vehicles = dict()

        if self.single_agent:
            return self._compute_observations()[self.trafficstate[0]]
        else:
            return self._compute_observations()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return self.sumo.simulation.getTime()

    def step(self, action):
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            for _ in range(self.min_SDHE):
                self._sumo_step()
        else:
            self._apply_actions(action)
            self._run_steps()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        self._compute_info()

        if self.single_agent:
            return observations[self.trafficstate[0]], rewards[self.trafficstate[0]], dones['__all__'], {}
        else:
            return observations, rewards, dones, {}

    def _run_steps(self):
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            for ts in self.trafficstate:
                self.Drivingbehavior[ts].update()
                if self.Drivingbehavior[ts].time_to_act:
                    time_to_act = True

    def _apply_actions(self, actions):

        if self.single_agent:
            if self.Drivingbehavior[self.trafficstate[0]].time_to_act:
                self.Drivingbehavior[self.trafficstate[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                if self.Drivingbehavior[ts].time_to_act:
                    self.Drivingbehavior[ts].set_next_phase(action)

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.trafficstate}
        dones['__all__'] = self.sim_step > self.sim_max_time
        return dones
    
    def _compute_info(self):
        info = self._compute_step_info()
        self.metrics.append(info)

    def _compute_observations(self):
        self.observations.update({ts: self.Drivingbehavior[ts].compute_observation() for ts in self.trafficstate if self.Drivingbehavior[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.Drivingbehavior[ts].time_to_act}

    def _compute_rewards(self):
        self.rewards.update({ts: self.Drivingbehavior[ts].compute_reward() for ts in self.trafficstate if self.Drivingbehavior[ts].time_to_act})
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.Drivingbehavior[ts].time_to_act}

    @property
    def observation_space(self):
        return self.Drivingbehavior[self.trafficstate[0]].observation_space
    
    @property
    def action_space(self):
        return self.Drivingbehavior[self.trafficstate[0]].action_space
    
    def observation_spaces(self, ts_id):
        return self.Drivingbehavior[ts_id].observation_space
    
    def action_spaces(self, ts_id):
        return self.Drivingbehavior[ts_id].action_space

    def _sumo_step(self):
        self.sumo.simulationStep()

    def _compute_step_info(self):
        return {
            'step_time': self.sim_step,
            'reward': self.Drivingbehavior[self.trafficstate[0]].last_reward,
            'total_stopped': sum(self.Drivingbehavior[ts].get_total_queued() for ts in self.trafficstate),
            'total_wait_time': sum(sum(self.Drivingbehavior[ts].get_waiting_time_per_lane()) for ts in self.trafficstate)
        }

    def close(self):
        if self.sumo is None:
            return
        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()
        try:
            self.disp.stop()
        except AttributeError:
            pass
        self.sumo = None
    
    def __del__(self):
        self.close()
    
    def render(self, mode='human'):
        if self.virtual_display:
            img = self.disp.grab()
            if mode == 'rgb_array':
                return np.array(img)
            return img         
    
    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + '_conn{}_run{}'.format(self.label, run) + '.csv', index=False)

    # Below functions are for discrete state space

    def encode(self, state, ts_id):
        phase = int(np.where(state[:self.Drivingbehavior[ts_id].traffic_phases] == 1)[0])
        min_green = state[self.Drivingbehavior[ts_id].traffic_phases]
        density_queue = [self._discretize_density(d) for d in state[self.Drivingbehavior[ts_id].traffic_phases + 1:]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase, min_green] + density_queue)

    def _discretize_density(self, density):
        return min(int(density*10), 9)


class SumoEnvironmentPZ(AECEnv, EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array'], 'name': "sumo_rl_v0"}

    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = SumoEnvironment(**self._kwargs)

        self.agents = self.env.trafficstate
        self.possible_agents = self.env.trafficstate
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}

        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.dones = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self):
        self.env.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        obs = self.env.observations[agent].copy()
        return obs

    def state(self):
        raise NotImplementedError('Method state() currently not implemented.')

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)
    
    def save_csv(self, out_csv_name, run):
        self.env.save_csv(out_csv_name, run)

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception('Action for agent {} must be in Discrete({}).'
                            'It is currently {}'.format(agent, self.action_spaces[agent].n, action))

        self.env._apply_actions({agent: action})

        if self._agent_selector.is_last():
            self.env._run_steps()
            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            self.env._compute_info()
        else:
            self._clear_rewards()
        
        done = self.env._compute_dones()['__all__']
        self.dones = {a : done for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
