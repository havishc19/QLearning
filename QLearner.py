"""  		   	  			    		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			    		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			    		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			    		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			    		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			    		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			    		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			    		  		  		    	 		 		   		 		  
or edited.  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			    		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			    		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			    		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Student Name: Havish Chennamraj (replace with your name)  		   	  			    		  		  		    	 		 		   		 		  
GT User ID: hchennamraj3 (replace with your User ID)  		   	  			    		  		  		    	 		 		   		 		  
GT ID: 903201642 (replace with your GT ID)  		   	  			    		  		  		    	 		 		   		 		  
"""  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
import numpy as np  		   	  			    		  		  		    	 		 		   		 		  
import random as rand  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
class QLearner(object):  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):
        self.smallNum = 0.000000001
        self.numStates = num_states
        self.numActions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.Q = np.random.uniform(-1.0, 1.0, [num_states,num_actions])
        self.s = 0
        self.a = 0

        self.experienceTuples = []
        self.T = np.full((self.numStates, self.numActions, self.numStates), self.smallNum)
        self.Tc = np.full((self.numStates, self.numActions, self.numStates), self.smallNum)
        self.Rewards = np.zeros((self.numStates, self.numActions))

    def author(self):
      return "hchennamraj3"


    def updateModel(self, s_prime,r):
        self.Tc[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] + 1
        self.T[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] / np.sum(self.Tc[self.s, self.a, :]) 
        retentionRate = 1 - self.alpha
        self.Rewards[self.s, self.a] = retentionRate * (self.Rewards[self.s, self.a]) + (self.alpha * r)

    def updateTable(self, s, a, s_prime, r):
      self.Q[s,a] = (1 - self.alpha) * (self.Q[s,a]) + self.alpha * ( r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime,:])])

    def DynaQ(self):
        for i in range(0,self.dyna):
          randomState = rand.randint(0, self.numStates-1)
          randomAction = rand.randint(0, self.numActions - 1)
          # newState = np.argmax(self.T[randomState, randomAction, :])
          # newState = np.argmax(np.random.multinomial(1, self.T[randomState,randomAction,:]))
          newState = rand.randint(0, self.numStates-1)
          reward = self.Rewards[randomState, randomAction]
          self.updateTable(randomState, randomAction, newState, reward)
            # self.Q[randomState, randomAction] = (1 - self.alpha) * (self.Q[randomState, randomAction]) + self.alpha * ( reward + self.gamma * self.Q[ newState,np.argmax(self.Q[newState,:])])

    def DynaQ1(self):
        for i in range(self.dyna):
          randomIndex = rand.randint(0,len(self.experienceTuples)-1)
          randomExperienceTuple = self.experienceTuples[randomIndex]
          self.updateTable(randomExperienceTuple[0], randomExperienceTuple[1], randomExperienceTuple[2], randomExperienceTuple[3])

    def querysetstate(self, s):  		   	  			    		  		  		    	 		 		   		 		  
        """  		   	  			    		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table  		   	  			    		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			    		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			    		  		  		    	 		 		   		 		  
        """  		   	  			    		  		  		    	 		 		   		 		  
        self.s = s  
        randomNum = np.random.uniform(0,1)
        if(randomNum < self.rar):		   	  			    		  		  		    	 		 		   		 		  
          action = rand.randint(0, self.numActions-1)
        else:
          action =np.argmax(self.Q[s,:])
        self.rar = self.rar * self.radr
        if self.verbose: print "s =", s,"a =",action  		   	  			    		  		  		    	 		 		   		 		  
        return action  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    def query(self,s_prime,r):  		   	  			    		  		  		    	 		 		   		 		  
        """  		   	  			    		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			    		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			    		  		  		    	 		 		   		 		  
        @param r: The ne state  		   	  			    		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			    		  		  		    	 		 		   		 		  
        """
        self.updateTable(self.s, self.a, s_prime, r)

        if(self.dyna > 0):
          # self.updateModel(s_prime, r)
          self.experienceTuples.append([self.s,self.a,s_prime,r])
          self.DynaQ1()
          # self.DynaQ()

        action = self.querysetstate(s_prime)
        self.a = action
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r  		   	  			    		  		  		    	 		 		   		 		  
        return action  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			    		  		  		    	 		 		   		 		  
    print "Remember Q from Star Trek? Well, this isn't him"  		   	  			    		  		  		    	 		 		   		 		  
