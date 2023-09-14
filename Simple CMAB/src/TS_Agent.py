# Importing required libs
import numpy as np

class TS_Agent:
    
    def __init__(self, agent_name, resource_required, instants, alpha):
        #Agent's initializaion: 
        # agent_name: Agent's name
        # resource_required: Amount of desired resource by the agent (in terms of number of instants)
        # instants: Total number of decision instants 
        self.__agent_name = agent_name
        self.__resource_required = resource_required
 
        #Thompsong Sampling learning algorithm parameters:
        # theta: Vector of unknown variable corresponding to each decision instant
        # mu: Vetor of estimated mean of each element in the theta vector
        # tau: Vector of standard deviation in each element in the theta vector
        # n: Vector of number of times each element (arm) of theta vector has been played
        # Q: Vector of observed Q-values of each element in the theta vector
        # alpha: learning parameter of the Thompson Sampling algorithm
        self.__theta = np.ones(instants,dtype=float)
        self.__mu = np.array([0]*len(self.__theta),dtype=float)
        self.__tau = np.array([0.5]*len(self.__theta),dtype=float)
        self.__n = np.array([0]*len(self.__theta),dtype=int)
        self.__Q = np.array([0]*len(self.__theta),dtype=float)
        self.__alpha = alpha
        
  
    #Setter methods:
    # Setter for the agent's name    
    def set_agent_name(self, agent_name):
        self.__agent_name = agent_name
        
    # Setter for the agent's required resource    
    def set_resource_required(self, resource_required):
        self.__resource_required = resource_required
    
    # Setter for the agent's theta vector 
    def set_theta(self, theta):
        self.__theta = theta
    
    # Setter for the agent's mu vector 
    def set_ag_name(self, mu):
        self.__mu = mu
     
    # Setter for the agent's tau vector 
    def set_ag_name(self, tau):
        self.__tau = tau
    
    # Setter for the agent's n vector
    def set_n(self, n):
        self.__n = n
     
    # Setter for the agent's Q vector
    def set_Q(self, Q):
        self.__Q = Q
     
    # Setter for the agent's learning paramter alpha 
    def set_alpha(self, alpha):
        self.__alpha = alpha
        

    
    #Getter methods:
    # Getter for the agent's name    
    def get_agent_name(self):
        return self.__agent_name 
        
    # Getter for the agent's required resource    
    def get_resource_required(self):
        return self.__resource_required
    
    # Getter for the agent's theta vector 
    def get_theta(self):
        return self.__theta
    
    # Getter for the agent's mu vector 
    def get_ag_name(self):
        return self.__mu 
     
    # Getter for the agent's tau vector 
    def get_ag_name(self):
        return self.__tau 
    
    # Getter for the agent's n vector
    def get_n(self):
        return self.__n
     
    # Getter for the agent's Q vector
    def get_Q(self):
        return self.__Q
     
    # Getter for the agent's learning paramter alpha 
    def get_alpha(self):
        return self.__alpha
    
    

    #Method to select estimated optimal actions         
    def select_actions(self):
        self.__selected_actions = np.argsort(-self.__theta)[0:self.__resource_required]
        
    #Getter for the selected estimated optimal actions       
    def get_selected_actions(self):        
        return self.__selected_actions
    
    
    #Update estimated value of the t-th element of the unkown theta vector:
    # t: t-th element of the unknwon theta vector
    # reward: observed reward for playing t-th element (arm) of theta vector
    def update_estimate(self, t, reward):
        self.__n[t] = self.__n[t] + 1
        self.__tau[t] = self.__tau[t] + self.__alpha * self.__n[t]
        self.__Q[t] = self.__Q[t] + reward
        self.__mu[t] = ((self.__tau[t]*self.__mu[t]) + (self.__Q[t]*self.__alpha))/(self.__tau[t] + self.__alpha*self.__n[t])
        self.__theta[t] = np.mean(np.random.normal(self.__mu[t], (1/self.__tau[t]), 1000))