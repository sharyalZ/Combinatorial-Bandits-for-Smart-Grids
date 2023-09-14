# Importing required libs
import numpy as np
import random

# Line Agent's Class
class Line_Agent:

    # Initializing the agent
    def __init__(self, name, rated_current, daily_price):
        self.name = name
        self.delta = 0.00
        self.rated_current = rated_current - self.delta
        self.price_data = daily_price

    # Method to assign instataneous rewards to EV agents
    def reward_evs(self, t, current, ev_agents):
    # If there is no congestion in the network
        if(current <= self.rated_current):
            for a in range(0,len(ev_agents)):
                ev_agents[a].cong = False
                if (t in ev_agents[a].selected_actions or t in np.where(ev_agents[a].solar_estimate>=0.2)[0]):
                    reward = (1 - (self.price_data [t]/np.max(self.price_data)))
                    ev_agents[a].update_estimate(t,reward)
                if (t in np.where(ev_agents[a].theta_solar>=0.2)[0]):
                    ev_agents[a].this_reward = 1
    # If there is congestion in the network                
        else:
            for a in range(0,len(ev_agents)):
                ev_agents[a].cong = True
            col_agents = []
            # The set of EV agents that are charging simultaneously
            for c1 in range(0,len(ev_agents)):
                if (t in ev_agents[c1].selected_actions or t in np.where(ev_agents[c1].solar_estimate>=0.2)[0]):
                    col_agents.append(ev_agents[c1])
            col_agents = np.array(col_agents)
            # Randomly sampling winner agents 
            random_pick = np.array(random.sample(range(len(col_agents)), int(np.floor(self.rated_current/(current/len(col_agents)))))) if (len(col_agents)>0) else np.array([], dtype = int)
            # Rewarding the uniformly sampled EVs
            for rp in range(0,len(random_pick)):
                reward = (1 - (self.price_data [t]/np.max(self.price_data)))
                col_agents[random_pick[rp]].update_estimate(t,reward)
            # Rewarding the remaining EVs that were not uniformly sampled    
            for a in range(0,len(col_agents)):
                if (((t in col_agents[a].selected_actions) or t in np.where(col_agents[a].solar_estimate >= 0.2)[0]) and not(a in random_pick)):
                    reward = -1
                    col_agents[a].update_estimate(t,reward)