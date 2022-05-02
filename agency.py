'''
Author:         David Radke
Affiliation:    University of Waterloo
Date:           May 2, 2022
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import os
import pickle
import utils
import copy
import scipy.stats

from datetime import datetime


import environment
import networks


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

class Population(object):
    def __init__(self, population_arr, teams, parameters):
        self._population = population_arr
        self._teams = teams
        self._summaryLength = parameters.summaryLength
        self._parameters = parameters

        game_str = utils.getGame(parameters)

        if self._parameters.sample_strategy:
            path = "self"+str(parameters.selfSS)+"team"+str(parameters.teamSS)+"sys"+str(parameters.systemSS)

            if parameters.ppo:
                self._savedir = game_str+"ppo/HETERO/"+path+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.mean_team:
                self._savedir = game_str+"mean_team/HETERO/"+path+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.per:
                self._savedir = game_str+"per/HETERO/"+path+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.beta_dist:
                self._savedir = game_str+"beta_"+str(parameters.beta_a)+"-"+str(parameters.nteams)+"_teams/"+str(parameters.beta_b)+"/HETERO/"+path+"/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.ninc > 0 and parameters.even_inc:
                self._savedir = game_str+"inc/even/"+str(parameters.ninc)+"-inc-agents/"+str(parameters.inc_deg)+"deg/HETERO/"+path+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.ninc > 0 and not parameters.even_inc:
                self._savedir = game_str+"inc/"+str(parameters.ninc)+"-inc-agents/"+str(parameters.inc_deg)+"deg/HETERO/"+path+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            else:
                self._savedir = game_str+"HETERO/"+path+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"

            # edit for plays per episode
            if parameters.plays_per_episode > 1:
                self._savedir = self._savedir + str(parameters.plays_per_episode) + "plays/"+str(parameters.nepisodes)+"episodes/c"+str(parameters.cost)+"b"+str(parameters.benefit)+"/"
            else:
                self._savedir = self._savedir + str(parameters.nepisodes)+"episodes/c"+str(parameters.cost)+"b"+str(parameters.benefit)+"/"

            # if parameters.team_sample_weight != 1:
            self._savedir = self._savedir + "team-weight-"+str(parameters.team_sample_weight)+"/"
        else:
            if parameters.a2c:
                self._savedir = "populations/a2c/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"+str(parameters.nepisodes)+"episodes/"
            elif parameters.pd and not parameters.pd_donation:
                self._savedir = game_str+"self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"+str(parameters.nepisodes)+"episodes/"
            # elif parameters.pd and parameters.pd_donation:
            else:
                if parameters.ppo:
                    self._savedir = game_str+"ppo/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
                elif parameters.mean_team:
                    self._savedir = game_str+"mean_team/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
                elif parameters.per:
                    self._savedir = game_str+"per/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
                elif parameters.beta_dist:
                    self._savedir = game_str+"beta_"+str(parameters.beta_a)+"-"+str(parameters.beta_b)+"/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
                elif parameters.ninc > 0 and parameters.even_inc:
                    self._savedir = game_str+"inc/even/"+str(parameters.ninc)+"-inc-agents/"+str(parameters.inc_deg)+"deg/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
                elif parameters.ninc > 0 and not parameters.even_inc:
                    self._savedir = game_str+"inc/"+str(parameters.ninc)+"-inc-agents/"+str(parameters.inc_deg)+"deg/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
                else:
                    self._savedir = game_str+"HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"

                # edit for plays per episode
                if parameters.plays_per_episode > 1:
                    self._savedir = self._savedir + str(parameters.plays_per_episode) + "plays/"+str(parameters.nepisodes)+"episodes/c"+str(parameters.cost)+"b"+str(parameters.benefit)+"/"
                else:
                    self._savedir = self._savedir +str(parameters.nepisodes)+"episodes/c"+str(parameters.cost)+"b"+str(parameters.benefit)+"/"

            self._savedir = self._savedir + "team-weight-"+str(parameters.team_sample_weight)+"/"


            if self._parameters.cd: self._savedir = self._savedir+"coordistribution/sigma-"+str(self._parameters.sigma)+"/"
            elif self._parameters.ohc: self._savedir = self._savedir+"oneHotCoordination/"
            elif self._parameters.nactions > self._parameters.nteams and not parameters.pd: self._savedir = self._savedir+str(self._parameters.nactions)+"action-sim/"

        _ = utils.checkPath(self._savedir)

        for a in self._population:
            _ = utils.checkPath(self._savedir+"team-"+str(a._team_num)+"/")
            utils.writeCSVTitle(self._savedir+"team-"+str(a._team_num)+"/"+str(a._id)+".csv", parameters.nteams)



    def checkBufferStats(self):
        exp_len = []
        for t in list(self._teams.keys()): # loop through teams
            for a in self._teams[t]._team: # loop through agents of a team

                exp_len.append(len(a._experience['s']))

        exp_len = np.array(exp_len)
        return mean_confidence_interval(exp_len)


    def avgAgentsRewards(self, agents_to_avg):
        rewards = []
        for a in agents_to_avg:
            rewards.append(np.mean(a._reward_lst,axis=1))

        rewards = np.array(rewards)
        sums = rewards.sum(axis=0)
        reward_avgs = sums/len(agents_to_avg)

        start = 0
        end = self._summaryLength #50
        step = 1
        avg_rewards_window = []
        while end < reward_avgs.shape[0]:
            avg_rewards_window.append(np.mean(reward_avgs[start:end]))
            start+=step
            end+=step
        return avg_rewards_window

    def getCooperationPercentage(self, agents):
        cooperation_percentages = {}
        cooperating = 0
        for a in agents: # loop through agents of a team
            recent_actions = a._action_mtx[:,(-1*self._parameters.endSummaryLength):,:]
            # print("recent: ", recent_actions.shape)
            recent_actions = recent_actions[recent_actions>=0]
            cooperation = np.count_nonzero(recent_actions == 0) / recent_actions.shape[0]             # assuming 0 is cooperation action
            cooperation_percentages[a._id] = cooperation

            if cooperation >= .5:
                cooperating+=1
        return cooperation_percentages, (cooperating/len(agents))


    def calculateLocalCooperation(self):
        if self._parameters.nagents == 1 or self._parameters.team_sample_weight == 0.0:
            self._N_l_arr = np.ones((self._parameters.nteams))*-1
            self._N_l = -1
            return

        local_coops = []
        for team_count, team in enumerate(self._all_agents_policy):
            for agent_count, agent in enumerate(team):
                towards_own_team = copy.deepcopy(self._all_agents_policy[team_count,:,team_count])
                towards_own_team[agent_count] = -1
                towards_own_team = towards_own_team[towards_own_team >= 0]
                local_coops.append(np.count_nonzero(towards_own_team == 0) / towards_own_team.size)
        self._N_l_arr = np.array(local_coops)                       # NOTE: 1 in this array is GOOD, means cooperation
        self._N_l = np.mean(self._N_l_arr)
        return

    def calculateGlobalCooperation(self):
        global_coops = []
        for team_count, team in enumerate(self._all_agents_policy):
            for agent_count, agent in enumerate(team):
                towards_my_team = copy.deepcopy(self._all_agents_policy[:,:,team_count])
                towards_my_team[team_count,agent_count] = -1
                towards_my_team = towards_my_team[towards_my_team >= 0]
                global_coops.append(np.count_nonzero(towards_my_team == 0) / towards_my_team.size)
        self._N_g_arr = np.array(global_coops)
        self._N_g = np.mean(self._N_g_arr)
        return

    def saveAgentObjects(self):
        for a in self._population:
            a._trainNet = None
            a._targetNet = None
            fname = a._polsavedir+a._id+".pkl"
            with open(fname, 'wb') as output:
                pickle.dump(a, output, pickle.HIGHEST_PROTOCOL)
        print("Agents Saved")

    def logSummary(self, epi, env):
        for t in list(self._teams.keys()):
            for a in self._teams[t]._team:
                for team in range(len(list(self._teams.keys()))):
                    unique, counts = np.unique(a._action_history[team], return_counts=True)
                    counts = counts / np.sum(counts)
                    if counts.size > 0:
                        a._majority_actions[team] = {unique[np.argmax(counts)]:np.around(np.amax(counts), decimals=3)}
                        a._action_frequency[team] = dict(zip(unique, np.around(counts, decimals=3)))
                    else:
                        a._majority_actions[team] = {-1:-1}
                        a._action_frequency[team] = {-1:-1}
                utils.writeCSVRow(self._savedir+"team-"+str(t)+"/"+str(a._id)+".csv", epi, a, self._parameters)



    def printSummary(self, epi, env):
        print("======== EPISODE ", epi, " SUMMARY ========")
        print("Epsilon: ", env._epsilon)
        for t in list(self._teams.keys()):
            print("TEAM: ", t)
            for a in self._teams[t]._team:
                for team in range(len(list(self._teams.keys()))):
                    unique, counts = np.unique(a._action_history[team], return_counts=True)
                    counts = counts / np.sum(counts)
                    if counts.size > 0:
                        a._majority_actions[team] = {unique[np.argmax(counts)]:np.around(np.amax(counts), decimals=3)}
                        a._action_frequency[team] = dict(zip(unique, np.around(counts, decimals=3)))
                    else:
                        a._majority_actions[team] = {-1:-1}
                        a._action_frequency[team] = {-1:-1}

                print(a, " - REWARD:", round(np.mean(np.mean(a._reward_lst[-self._summaryLength:,:],axis=1)),3), " HISTORY: ", a._majority_actions)

    def loadFeatures(self, parameters):
        dir = self._savedir+"policies/"
        self._parameters = parameters

        self._all_agents_policy = np.load(dir+"full-policy-mtx.npy")
        self.calculateLocalCooperation()
        self.calculateGlobalCooperation()
        self._coop_ratio = self._N_l / self._N_g


    def avgAgentsCoop(self, parameters):
        dir = self._savedir+"policies/"

        total_coop_percentage = []
        for a in self._population:
            for team_count in range(parameters.nteams):
                if (parameters.nagents == 1 or self._parameters.team_sample_weight == 0.0) and (team_count == a._team_num): continue
                coop = np.load(dir+a._id+"policies/towards_team-"+str(team_count)+".npy")
                total_coop_percentage.append(np.mean(np.array(coop[0,-25000:])))
        return np.mean(np.array(total_coop_percentage))


    def savePolicies(self):
        ''' Saves representations of the policies, including:
        policy-mtx.npy = (nteams, nteams) numpy array, row team plays [row,col] action against col team
        towards_team-[TEAM].npy = (nactions, nepisodes-end) listing the frequency of each action taken when playing each team
        '''
        print("Saving Policies")

        dir = self._savedir+"policies/"
        _ = utils.checkPath(dir)

        policy = np.ones((self._parameters.nteams, self._parameters.nteams))*-1
        self._all_agents_policy = np.ones((self._parameters.nteams, self._parameters.nagents, self._parameters.nteams))*-1
        for agentteam, t in enumerate(list(self._teams.keys())):
            team_policy = np.ones((self._parameters.nagents, self._parameters.nteams))*-1
            for agent_count, a in enumerate(self._teams[t]._team):
                for team in range(len(list(self._teams.keys()))):
                    unique, counts = np.unique(a._action_history[team][(-1*self._parameters.endSummaryLength):], return_counts=True)
                    counts = counts / np.sum(counts)
                    if counts.size > 0:
                        team_policy[agent_count,team] = unique[np.argmax(counts)]
                    else:
                        team_policy[agent_count,team] = -1
                self._all_agents_policy[agentteam,agent_count] = team_policy[agent_count]
            policy[agentteam] = np.mean(team_policy, axis=0)
        np.save(dir+"policy-mtx.npy", policy)
        np.save(dir+"full-policy-mtx.npy", self._all_agents_policy)

        all_agents_policy = copy.deepcopy(self._all_agents_policy)
        all_agents_policy = all_agents_policy.reshape((self._parameters.nagents*self._parameters.nteams),self._parameters.nteams)
        pd.DataFrame(all_agents_policy).to_csv(dir+"full-policy-mtx.txt", index=False, header=None)
        pd.DataFrame(policy).to_csv(dir+"policy-mtx.txt", index=False, header=None)

        self.calculateLocalCooperation()
        self.calculateGlobalCooperation()
        if self._N_g != 0:
            self._coop_ratio = self._N_l / self._N_g
        else:
            self._coop_ratio = 0

        ''' saving policy towards a team '''
        for t in list(self._teams.keys()): # loop through teams
            for a in self._teams[t]._team: # loop through agents of a team

                a._experience = a._trainNet.experience

                utils.checkPath(dir+a._id+"policies/")
                for team_count, team in enumerate(a._action_mtx): # history of actions towards each team
                    start, end, step, idx = 0, self._summaryLength, 1, 0 #self._summaryLength
                    actions_towards_team = np.zeros((self._parameters.nactions,(self._parameters.nepisodes-end)))
                    if (self._parameters.nagents == 1 or self._parameters.team_sample_weight == 0.0) and (a._team_num == team_count): continue        # hack for 1-agent teams
                    while end < a._action_mtx.shape[1]:
                        slice = a._action_mtx[team_count,start:end,:]

                        for action in range(self._parameters.nactions):
                            perc = (np.count_nonzero(slice == action) / np.count_nonzero(slice >= 0))
                            actions_towards_team[action,idx] = perc
                        idx+=1
                        start+=step
                        end+=step

                    np.save(dir+a._id+"policies/towards_team-"+str(team_count)+".npy", actions_towards_team)

    def calculateOutcomes(self):
        '''
        This function calculates the "outcomes" of an agent to determine how their policy evolves over time
        (i.e., do they learn defection because they are playing against more defection)

        # actions: outcome_labels = ['MCoop', 'Sucker', 'Tempt', 'MDef']

        output: saves images to ~/*-*policies/outcomes/....
        '''
        print("Calculating Outcomes")

        outcome_labels = ['ID', 'MCoop', 'Sucker', 'Tempt', 'MDef']
        df = pd.DataFrame(columns=outcome_labels)

        dir = self._savedir+"policies/"
        _ = utils.checkPath(dir)

        count = 0
        for t in list(self._teams.keys()): # loop through teams
            for a in self._teams[t]._team: # loop through agents of a team
                outcome_count = {"mcoop": [],
                                "sucker": [],
                                "tempt": [],
                                "mdef": []}

                # utils.checkPath(dir+a._id+"policies/outcomes/")
                for team_count, team in enumerate(a._outcomes): # history of actions towards each team

                    outcomes_towards_team = np.load(dir+a._id+"policies/outcomes/outcomes_towards_team-"+str(team_count)+".npy")

                    if team_count != a._team_num:
                        for action_idx, action_results in enumerate(outcomes_towards_team):
                            key = list(outcome_count.keys())[action_idx]
                            outcome_count[key].append(np.mean(action_results[:10000]))

                to_add = [a._id]
                for key in list(outcome_count.keys()):
                    outcome_count[key] = np.mean(np.array(outcome_count[key]))
                    to_add.append(np.mean(np.array(outcome_count[key])))
                df.loc[count] = to_add
                count+=1
        df = df.sort_values('MCoop', ascending=False)


    def saveNetworks(self):

        dir = self._savedir+"policies/"
        _ = utils.checkPath(dir)

        for a in self._population:
            a._trainNet.model.save(dir+a._id+"policies/"+a._id+"-trainNet")
            a._targetNet.model.save(dir+a._id+"policies/"+a._id+"-targetNet")

        print("Models saved")



class Team(object):
    def __init__(self, team_num, agent_lst, strategy=None, parameters=None):
        self._team_num = team_num
        self._team = agent_lst
        self._team_reward = {}
        self._team_games = {}

        if strategy is None:
            self._strategy = np.array([parameters.selfW, parameters.teamW, parameters.systemW])
        else:
            self._strategy = strategy
        self._team_q = np.zeros((parameters.nteams,parameters.nactions,parameters.nagents))

    def __repr__(self):
        return "TEAM(id: {}, Roster size: {}, SelfW: {}, TeamW: {}, SystemW: {})".format(self._team_num, len(self._team), self._strategy[0], self._strategy[1], self._strategy[2])

class Agent(object):
    def __init__(self, id, team, parameters, strategy=None):
        self._id = id
        self._team = np.zeros(parameters.nteams)
        self._team[team] = 1
        self._team_num = team
        hidden_units = [200, 200]

        ''' track actions taken towards a specific team '''
        action_amt = parameters.nactions
        self._action_mtx = np.ones((parameters.nteams, parameters.nepisodes, parameters.plays_per_episode)) * -1
        self._q_vals = np.ones((parameters.nteams, parameters.nactions, parameters.nepisodes, parameters.plays_per_episode)) * -1

        self._outcomes = np.ones((parameters.nteams, parameters.nepisodes, (parameters.nagents)+1)) * -1 # idx 0 will be when agent is focal, then any others will be when they are opponent

        self._action_history = {}
        self._action_frequency = {}
        self._majority_actions = {}
        for i in range(parameters.nteams):
            self._action_history[i] = []
            self._action_frequency[i] = {}
            self._majority_actions[i] = {}
            for j in range(action_amt):
                self._action_frequency[i][j] = 0


        self._reward_lst = np.zeros((parameters.nepisodes, parameters.plays_per_episode)) #[]
        self._rewards = 0
        self._iter = 0
        self._done = False
        self._losses = list()
        self._copy_step = parameters.copy_step
        self._coordistribution = parameters.cd
        self._one_hot_coordination = parameters.ohc
        self._eig_means = []
        self._eig_stds = []
        self._save_eig_hist = parameters.save_eig_hist
        self._len_buffer = []
        self._inc = False # is agent incompetent

        ''' reward weights for prisoners dilemma '''
        if strategy is None:
            self._selfW = parameters.selfW
            self._teamW = parameters.teamW
            self._systemW = parameters.systemW
        else:
            self._selfW = strategy[0]
            self._teamW = strategy[1]
            self._systemW = strategy[2]

        self._team_sample_weight = parameters.team_sample_weight

        game_str = utils.getGame(parameters)

        if parameters.sample_strategy:
            path = "self"+str(parameters.selfSS)+"team"+str(parameters.teamSS)+"sys"+str(parameters.systemSS)

            if parameters.ppo:
                self._polsavedir = game_str+"ppo/HETERO/"+path+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.mean_team:
                self._polsavedir = game_str+"mean_team/HETERO/"+path+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.per:
                self._polsavedir = game_str+"per/HETERO/"+path+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.beta_dist:
                self._polsavedir = game_str+"beta_"+str(parameters.beta_a)+"-"+str(parameters.beta_b)+"/HETERO/"+path+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.ninc > 0 and parameters.even_inc:
                self._polsavedir = game_str+"inc/even/"+str(parameters.ninc)+"-inc-agents/"+str(parameters.inc_deg)+"deg/HETERO/"+path+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.ninc > 0 and not parameters.even_inc:
                self._polsavedir = game_str+"inc/"+str(parameters.ninc)+"-inc-agents/"+str(parameters.inc_deg)+"deg/HETERO/"+path+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            else:
                self._polsavedir = game_str+"HETERO/"+path+"/"+str(parameters.nteams)+"_teams/"+"/"+str(parameters.nagents)+"-agent-teams/"

            # edit for plays per episode
            if parameters.plays_per_episode > 1:
                self._polsavedir = self._polsavedir + str(parameters.plays_per_episode) + "plays/"+str(parameters.nepisodes)+"episodes/c"+str(parameters.cost)+"b"+str(parameters.benefit)+"/"
            else:
                self._polsavedir = self._polsavedir + str(parameters.nepisodes)+"episodes/c"+str(parameters.cost)+"b"+str(parameters.benefit)+"/"

        else:
            if parameters.a2c:
                self._polsavedir = "populations/a2c/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"+str(parameters.nepisodes)+"episodes/"
            elif parameters.pd and not parameters.pd_donation:
                self._polsavedir = game_str+"self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"+str(parameters.nepisodes)+"episodes/"
            else:
                if parameters.ppo:
                    self._polsavedir = game_str+"ppo/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
                elif parameters.mean_team:
                    self._polsavedir = game_str+"mean_team/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
                elif parameters.per:
                    self._polsavedir = game_str+"per/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
                elif parameters.beta_dist:
                    self._polsavedir = game_str+"beta_"+str(parameters.beta_a)+"-"+str(parameters.beta_b)+"/HOMOG/"+"self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
                elif parameters.ninc > 0 and parameters.even_inc:
                    self._polsavedir = game_str+"inc/even/"+str(parameters.ninc)+"-inc-agents/"+str(parameters.inc_deg)+"deg/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
                elif parameters.ninc > 0 and not parameters.even_inc:
                    self._polsavedir = game_str+"inc/"+str(parameters.ninc)+"-inc-agents/"+str(parameters.inc_deg)+"deg/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
                else:
                    self._polsavedir = game_str+"HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"

                # edit for plays per episode
                if parameters.plays_per_episode > 1:
                    self._polsavedir = self._polsavedir + str(parameters.plays_per_episode) + "plays/"+str(parameters.nepisodes)+"episodes/c"+str(parameters.cost)+"b"+str(parameters.benefit)+"/"
                else:
                    self._polsavedir = self._polsavedir +str(parameters.nepisodes)+"episodes/c"+str(parameters.cost)+"b"+str(parameters.benefit)+"/"

        self._polsavedir = self._polsavedir + "team-weight-"+str(parameters.team_sample_weight)+"/"

        self._polsavedir = self._polsavedir + "policies/"+str(self._id)+"policies/"

        if not os.path.exists(self._polsavedir):
            os.makedirs(self._polsavedir)

        if parameters.a2c:
            self._trainNet = networks.A2C(parameters, hidden_units)
        else:
            self._trainNet = networks.DQN("train", self._id, parameters, hidden_units, self._polsavedir)
            self._targetNet = networks.DQN("target", self._id, parameters, hidden_units, self._polsavedir)

    def calcQVals(self, nactions, target_team_num):

        q_vals = []
        for action_idx in range(nactions): # loop through actions
            action_vals = self._q_vals[target_team_num,action_idx,-25000:,:]
            action_vals = action_vals[action_vals != -1]
            q_vals.append(np.mean(action_vals))
        return q_vals[0] - q_vals[1]

    def recordResult(self, focal_action, opp_action, epi, opponent, is_focal=False):
        ''' Records mutual cooperation (0),
                    sucker (1),
                    temptation (2),
                    mutual defection (3) '''
        # np.ones((parameters.nteams, parameters.nepisodes, ((parameters.nteams*parameters.nagents)+1)))

        # mutual cooperation
        if focal_action == 0 and opp_action == 0:
            if is_focal: self._outcomes[opponent._team_num, epi, 0] = 0
            else: self._outcomes[opponent._team_num, epi, int(opponent._id[-1])+1] = 0
        # sucker
        elif focal_action == 0 and opp_action == 1:
            if is_focal: self._outcomes[opponent._team_num, epi, 0] = 1
            else: self._outcomes[opponent._team_num, epi, int(opponent._id[-1])+1] = 1
        # temptation
        elif focal_action == 1 and opp_action == 0:
            if is_focal: self._outcomes[opponent._team_num, epi, 0] = 2
            else: self._outcomes[opponent._team_num, epi, int(opponent._id[-1])+1] = 2
        # mutual defection
        elif focal_action == 1 and opp_action == 1:
            if is_focal: self._outcomes[opponent._team_num, epi, 0] = 3
            else: self._outcomes[opponent._team_num, epi, int(opponent._id[-1])+1] = 3


    def __repr__(self):
        return "Agent(id: {}, team: {})".format(self._id, self._team_num)

def getNincArray(parameters):
    ninc_arr = np.zeros(parameters.nteams)
    to_turn_inc = copy.deepcopy(parameters.ninc)
    idx = 0
    while to_turn_inc > 0:
        if idx == ninc_arr.size:
            idx = 0
        ninc_arr[idx]+=1
        idx+=1
        to_turn_inc-=1
    return ninc_arr

def loadPopulation(population_fname):
    print("Loading: ", population_fname)
    with open(population_fname, "rb") as f:
        population_obj = pickle.load(f)

    agent_lst = population_obj._population
    teams = population_obj._teams
    return agent_lst, teams, population_obj

def createPopulation(parameters):
    print("Making Population...")

    game_str = utils.getGame(parameters)

    if parameters.sample_strategy:
        if parameters.ppo:
            dir = game_str+"ppo/HETERO/"
        elif parameters.mean_team:
            dir = game_str+"mean_team/HETERO/"
        elif parameters.per:
            dir = game_str+"per/HETERO/"
        elif parameters.ninc > 0 and parameters.even_inc:
            dir = game_str+str(parameters.ninc)+"inc/even/HETERO/"
        elif parameters.ninc > 0 and not parameters.even_inc:
            dir = game_str+str(parameters.ninc)+"inc/HETERO/"
        else:
            dir = game_str+"HETERO/"

        dir = dir+"self"+str(parameters.selfSS)+"team"+str(parameters.teamSS)+"sys"+str(parameters.systemSS)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"+str(parameters.nepisodes)+"episodes/c"+str(parameters.cost)+"b"+str(parameters.benefit)+"/"

    else:
        if parameters.a2c:
            dir = "populations/a2c/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"+str(parameters.nepisodes)+"episodes/"
        elif parameters.pd and not parameters.pd_donation:
            dir = game_str+"self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"+str(parameters.nepisodes)+"episodes/"
        # elif parameters.pd and parameters.pd_donation:
        else:
            if parameters.ppo:
                dir = game_str+"ppo/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.mean_team:
                dir = game_str+"mean_team/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.per:
                dir = game_str+"per/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.beta_dist:
                dir = game_str+"beta_"+str(parameters.beta_a)+"-"+str(parameters.beta_b)+"/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.ninc > 0 and parameters.even_inc:
                dir = game_str+"inc/even/"+str(parameters.ninc)+"-inc-agents/"+str(parameters.inc_deg)+"deg/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            elif parameters.ninc > 0 and not parameters.even_inc:
                dir = game_str+"inc/"+str(parameters.ninc)+"-inc-agents/"+str(parameters.inc_deg)+"deg/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"
            else:
                dir = game_str+"HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"+str(parameters.nteams)+"_teams/"+str(parameters.nagents)+"-agent-teams/"

            # edit for plays per episode
            if parameters.plays_per_episode > 1:
                dir = dir + str(parameters.plays_per_episode) + "plays/"+str(parameters.nepisodes)+"episodes/c"+str(parameters.cost)+"b"+str(parameters.benefit)+"/"
            else:
                dir = dir +str(parameters.nepisodes)+"episodes/c"+str(parameters.cost)+"b"+str(parameters.benefit)+"/"

        if parameters.cd: dir = dir+"coordistribution/sigma-"+str(parameters.sigma)+"/"
        elif parameters.ohc: dir = dir+"oneHotCoordination/"
        elif parameters.nactions > parameters.nteams and not parameters.pd: dir = dir+str(parameters.nactions)+"action-sim/"

    _ = utils.checkPath(dir)

    strategies = utils.makeStrategies(parameters)

    strategy = None
    agents = {}
    agent_lst = []
    teams = {}
    ninc_so_far = 0
    if parameters.even_inc:
        ninc_so_far_teams = getNincArray(parameters)

    for team in range(parameters.nteams):
        if parameters.even_inc: ninc_so_far = 0
        if parameters.sample_strategy:
            if team < parameters.selfSS:
                strategy = strategies[0]
            elif team < (parameters.selfSS+ parameters.teamSS):
                strategy = strategies[1]
            else:
                strategy = strategies[2]

        agents[team] = []
        for a in range(parameters.nagents):
            id = str(team) + "-" + str(a)
            if parameters.load_agents:
                agent_obj = utils.loadAgent(id, dir, parameters)
            else:
                agent_obj = Agent(id, team, parameters, strategy)

            # make rogue
            if parameters.even_inc:
                if ninc_so_far < ninc_so_far_teams[team]:
                    agent_obj._inc = True
                    ninc_so_far += 1
            else:
                if ninc_so_far < parameters.ninc and a != (parameters.nagents-1): # now this makes sure there is at least one competent agent on a team
                    agent_obj._inc = True
                    ninc_so_far += 1

            agents[team].append(agent_obj)
            agent_lst.append(agent_obj)

        teams[team] = Team(team, agents[team], strategy, parameters)
    population_obj = Population(agent_lst, teams, parameters)
    return agent_lst, teams, population_obj

def makePopulation(parameters):
    ''' Make Population of agents '''

    agent_lst, teams, population_obj = createPopulation(parameters)
    env = environment.TeamCooperation(parameters,agent_lst=agent_lst)
    return agent_lst, teams, env, population_obj
