# Importing required libs
import numpy as np

# Defining EV agent's class
class EV_Agent:

    def __init__(self, name, start_time, end_time, charge_required,rank, daily_price, tau, solar_data):
    # Defining variables of the EV agent
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.sort_param = (end_time-start_time)
        self.charge_required = charge_required
        self.rank = rank
        self.price_data = daily_price
        self.available_slots = np.linspace(0,1439,1440)
        self.taken_slots_message = np.array([])
        self.theta = np.zeros(1440,dtype=float)
        self.solar_data = solar_data

    # Defining variables related to CMAB learning algorithm
        self.ini = 0
        self.n = np.array([0]*len(self.theta),dtype=int)
        self.tau = np.array([tau]*len(self.theta),dtype=float)
        self.tau0 = np.array([2]*len(self.theta),dtype=float)
        self.mu0 = np.array([0]*len(self.theta),dtype=float)
        self.mu0 = 1- self.price_data
        self.Q = np.array([0]*len(self.theta),dtype=float)
        self.Q_ratio = np.array([0]*len(self.theta),dtype=float)
        self.remaining_chrg = 0
        
    # Defining variables related to solar estimation
        self.theta_solar = np.zeros(1440,dtype=float)
        self.n_solar = np.array([0]*len(self.theta_solar),dtype=int)
        self.tau_solar = np.array([100]*len(self.theta_solar),dtype=float)
        self.tau0_solar = np.array([5]*len(self.theta_solar),dtype=float)
        self.mu0_solar = np.array([0]*len(self.theta_solar),dtype=float)
        self.Q_solar = np.array([0]*len(self.theta_solar),dtype=float)
        
    # Calling the methods to make estimations of solar vectors and unknown parameters vector    
        self.estimate_theta_solar()
        self.estimate_theta()


        self.inst_voltage = 1.0
        self.cong = False
        self.this_reward = 0
        self.solar_estimate = np.zeros(1440,dtype=float)
        self.reward_history = []
        self.reward_historyb = []

    # Method to estimate theta vector
    def estimate_theta_solar(self):
        for i in range(0,len(self.theta)):
            if ((i >= self.start_time) or (i <= self.end_time)):
                self.theta_solar[i] = np.mean(np.random.normal(self.solar_data[i], (1/self.tau0_solar[i]), 50))
            else:
                self.theta_solar[i] = np.mean(np.random.normal(0, (1/1000), 5))
        
    # Method to estimate unknown parameters vector
    def estimate_theta(self):
        for i in range(0,len(self.theta)):
            if ((i >= self.start_time) or (i <= self.end_time)):
                self.theta[i] = np.mean(np.random.normal(1-(self.price_data[i]/np.max(self.price_data)), (1/self.tau0[i]), 50))
            else:
                self.theta[i] = np.mean(np.random.normal(-10, (1/100), 5))

    # Method to selec actions
    def select_actions(self):
        self.solar_estimate = (self.theta > 0) * self.theta_solar
        self.remaining_chrg = max(int(np.ceil(self.charge_required - (np.sum(self.solar_estimate)/7))),0)
        if (self.ini == 0):
            ta = np.copy(self.price_data)
            for t in range(0,len(ta)):
                if ((t>= self.end_time) and (t<= self.start_time)):
                    ta[t] = 1000
            self.selected_actions = np.delete(np.argsort(ta), np.where(self.solar_estimate>=0.2)[0])[0:self.remaining_chrg]
            return self.selected_actions
        else:
            self.selected_actions = np.delete(np.argsort(-self.theta), np.where(self.solar_estimate>=0.2)[0])[0:self.remaining_chrg]
            return self.selected_actions

    # Method to update the estimate of the theta vector
    def update_estimate(self, t, reward):
        self.reward_historyb.append(reward)
        reward = self.voltage_filter(reward)
        self.reward_history.append(reward)
        self.ini = self.ini + 1
        self.n[t] = self.n[t] + 1
        self.tau0[t] = self.tau0[t] + self.tau[t]*self.n[t]
        self.Q[t] = self.Q[t] + reward
        self.Q_ratio[t] = self.Q[t]/self.n[t]
        self.mu0[t] = ((self.tau0[t]*self.mu0[t]) + (self.Q[t]*self.tau[t]))/(self.tau0[t] + self.tau[t]*self.n[t])
        self.theta[t] = np.mean(np.random.normal(self.mu0[t], (1/self.tau0[t]), 1000))
        
        
    # Method to update the estimate of the solar vector
    def update_solar_estimate(self, t, reward):
        self.n_solar[t] = self.n_solar[t] + 1
        self.tau0_solar[t] = self.tau0_solar[t] + self.tau_solar[t]*self.n_solar[t]
        self.Q_solar[t] = self.Q_solar[t] + reward
        self.mu0_solar[t] = ((self.tau0_solar[t]*self.mu0_solar[t]) + (self.Q_solar[t]*self.tau_solar[t]))/(self.tau0_solar[t] + self.tau_solar[t]*self.n_solar[t])
        self.theta_solar[t] = np.mean(np.random.normal(self.mu0_solar[t], (1/self.tau0_solar[t]), 1000))
        
    # Method to get the average reward    
    def get_avg_reward(self):
        return np.sum(self.Q_ratio[self.selected_actions])/len(self.selected_actions)


    # Method emulating bus agent's functionalities
    def voltage_filter(self,reward):
        f_reward = 0
        if (self.inst_voltage < 0.95):
            b_reward = -1
        if (self.inst_voltage > 1.05):
            b_reward = -1
        if (self.inst_voltage <= 1.05 and self.inst_voltage >=0.95):
            b_reward = 0
        if (b_reward == 0 or self.cong == True):
            f_reward = reward
        else:
            f_reward = b_reward

        return f_reward