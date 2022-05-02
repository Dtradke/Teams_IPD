'''
Author:         David Radke
Affiliation:    University of Waterloo
Date:           May 2, 2022
'''

import tensorflow as tf
import numpy as np
import copy
from scipy.stats import norm
from scipy.special import softmax



class TeamCooperation():
    def __init__(self, parameters, agent_lst):

        # self._payoff = np.identity(nactions)
        action_amt = parameters.nactions #max(parameters.nactions, parameters.nteams)
        self._payoff_mtx_base = np.zeros((action_amt, action_amt))
        self._coordinate_mtx = np.identity(action_amt)

        pd_max = 6
        self._pd = parameters.pd
        self._pd_donation = parameters.pd_donation
        self._parameters = parameters

        if self._pd_donation:
            self._pd_row = np.array([np.array([(parameters.benefit - parameters.cost), (-1*parameters.cost)]), np.array([parameters.benefit, 0])])
            self._pd_col = np.array([np.array([(parameters.benefit - parameters.cost), parameters.benefit]), np.array([(-1*parameters.cost), 0])])
        elif parameters.stag_hunt:
            self._pd_row = np.array([np.array([4, 1]), np.array([3, 2])])
            self._pd_col = np.array([np.array([4, 3]), np.array([1, 2])])
            # self._pd_row = np.array([np.array([(parameters.benefit), (-1*parameters.cost)]), np.array([(parameters.benefit - parameters.cost), 0])])
            # self._pd_col = np.array([np.array([(parameters.benefit), (parameters.benefit - parameters.cost)]), np.array([(-1*parameters.cost), 0])])
        elif parameters.hawk_dove:
            self._pd_row = np.array([np.array([((parameters.cost - parameters.benefit)/2), parameters.cost]), np.array([0, parameters.cost/2])])
            self._pd_col = np.array([np.array([((parameters.cost - parameters.benefit)/2), 0]), np.array([(parameters.cost), (parameters.cost/2)])])
        elif parameters.chicken:
            self._pd_row = np.array([np.array([0, (-1*parameters.cost)]), np.array([parameters.cost, (-1*patameters.benefit)])])
            self._pd_col = np.array([np.array([0, parameters.cost]), np.array([(-1*parameters.benefit), (-1*patameters.benefit)])])
        else:
            self._pd_row = np.array([np.array([3, 0]), np.array([5, 1])])
            self._pd_col = np.array([np.array([3, 5]), np.array([0, 1])])


        self._agent_lst = agent_lst
        self._nactions = action_amt
        self._nteams = parameters.nteams
        self._state = [0]
        self._episode_ended = False
        self._plays_per_episode = parameters.plays_per_episode

        self._epsilon = parameters.epsilon
        self._min_epsilon = parameters.min_epsilon
        self._sigma = parameters.sigma
        self._coordistribution = parameters.cd
        self._one_hot_coordination = parameters.ohc


    def reset(self):
        self._state = [0]
        self._episode_ended = False
        return

    def makeOneHotCoordination(self, agent, opponent, mtx):
        action = agent._team_num + (opponent._team_num+1)
        actions = np.arange(self._nactions)
        zeros = np.zeros(self._nactions)
        zeros[action] = 1
        mtx[actions,actions] = zeros
        return mtx

    def makeDistribution(self, agent, opponent, mtx):
        mu = agent._team_num + opponent._team_num
        sigma = self._sigma
        actions = np.arange(self._nactions)
        dist = norm.pdf(actions, mu, sigma)
        mtx[actions,actions] = dist
        mtx = (mtx - np.amin(mtx)) / (np.amax(mtx) - np.amin(mtx))
        mtx = (mtx*2) - 1
        return mtx

    def fillPayoffMtx(self, agent, opponent, type):
        if type == 'coordinate':
            mtx = copy.deepcopy(self._coordinate_mtx)
            if self._coordistribution:
                mtx = self.makeDistribution(agent, opponent, mtx)
            elif self._one_hot_coordination:
                mtx = self.makeOneHotCoordination(agent, opponent, mtx)
        else:
            mtx = copy.deepcopy(self._payoff_mtx_base)
            mtx[type,type] = 1
        mtx[mtx == 0] = -1
        return mtx, mtx


    def step(self, action1, action2, agent, opponent, play):
        if self._parameters.pd or self._parameters.pd_donation or self._parameters.stag_hunt or self._parameters.chicken or self._parameters.hawk_dove:
            row_mtx, col_mtx = self._pd_row, self._pd_col
        else:
            if agent._team_num == opponent._team_num:
                row_mtx, col_mtx = self.fillPayoffMtx(agent, opponent, agent._team_num)
            else:
                row_mtx, col_mtx = self.fillPayoffMtx(agent, opponent, 'coordinate')

        row_r = row_mtx[action1, action2]
        col_r = col_mtx[action1, action2]

        done = 0 # manually fix done to 0. If 1, agents always defect (one-shot)
        return row_r, col_r, done



    def sampleWeights(self, agent, population_obj):
        ''' Change the probability of sampling a counterpart to be a teammate '''
        if agent._team_sample_weight == 1:
            team_prob = 1 / len(list(population_obj._teams.keys()))
            return (team_prob/(population_obj._parameters.nagents-1)), (team_prob/(population_obj._parameters.nagents)) # in-team, out-team
        elif agent._team_sample_weight > 1:
            team_prob = 1 / (len(list(population_obj._teams.keys()))+(agent._team_sample_weight-1)) #+1
            return (agent._team_sample_weight*(team_prob/(population_obj._parameters.nagents-1))), (team_prob/(population_obj._parameters.nagents)) # in-team, out-team
        elif agent._team_sample_weight < 1:
            other_weight = 1/agent._team_sample_weight
            team_prob = 1 / (len(list(population_obj._teams.keys()))+((other_weight-1)*(len(list(population_obj._teams.keys()))-1)))
            return (team_prob/(population_obj._parameters.nagents-1)), (other_weight*(team_prob/(population_obj._parameters.nagents))) # in-team, out-team


    def sample(self, population_obj, agent):
        sample_agent_lst = []
        probs = []
        agent_idx = 0
        count = 0
        if population_obj._parameters.nagents > 1 and agent._team_sample_weight != 0:
            team_prob = 1 / len(list(population_obj._teams.keys()))

            in_team, out_team = self.sampleWeights(agent, population_obj)

            for t in list(population_obj._teams.keys()):
                for a in population_obj._teams[t]._team:
                    sample_agent_lst.append(a)
                    if a._id == agent._id:
                        probs.append(0)
                        agent_idx = count
                    elif a._team_num == agent._team_num:
                        probs.append(in_team)
                    else:
                        probs.append(out_team)
                    count+=1

        elif agent._team_sample_weight == 0.0:      # does not play own team
            for t in list(population_obj._teams.keys()):
                for a in population_obj._teams[t]._team:
                    sample_agent_lst.append(a)
                    if a._team_num == agent._team_num:
                        probs.append(0)
                    else:
                        probs.append((1/(population_obj._parameters.nteams-1)) / population_obj._parameters.nagents)

        else:
            for t in list(population_obj._teams.keys()):
                for a in population_obj._teams[t]._team:
                    sample_agent_lst.append(a)
                    if a._id == agent._id:
                        probs.append(0)
                    else:
                        probs.append(1/(population_obj._parameters.nteams-1))

        return np.random.choice(sample_agent_lst, p=probs)

def pickPartner(self, agents):
    return 0
