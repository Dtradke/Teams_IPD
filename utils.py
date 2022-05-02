'''
Author:         David Radke
Affiliation:    University of Waterloo
Date:           May 2, 2022
'''
import numpy as np
import os
import pickle
import sys
import csv
import networks
import plots


def checkPath(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True

def loadAgent(id, dir, parameters):
    dir = dir + "team-weight-"+str(parameters.team_sample_weight)+"/policies/"+id+"policies/"+id+".pkl"
    with open(dir, "rb") as f:
        agent = pickle.load(f)
    return agent


def makeStrategies(parameters):
    # hard coded extreme policies - assuming it goes: self, team, system
    strategies = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]

    return strategies


def writeCSVTitle(agent_fname, num_teams):
    arr = ['Episode','Reward','Team','Agents-Per-Team']
    for i in range(num_teams):
        arr.append("t"+str(i)+"a")
        arr.append("t"+str(i)+"aP")
    with open(agent_fname, 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(arr)

def writeCSVRow(agent_fname, epi, agent, params):
    arr = [epi, round(np.mean(np.array(agent._reward_lst[(-1*params.summaryLength):])),3), agent._team_num, params.nagents]

    for i in list(agent._majority_actions.keys()):
        maj_action_dict = agent._majority_actions[i]
        arr.append(list(maj_action_dict.keys())[0])
        arr.append(maj_action_dict[list(maj_action_dict.keys())[0]])

    with open(agent_fname, 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(arr)

def getGame(parameters):
    if parameters.pd and not parameters.pd_donation:
        return 'populations/prisoners/'
    elif parameters.pd and parameters.pd_donation:
        return 'populations/prisoners_donation/'
    elif parameters.stag_hunt:
        return 'populations/stag_hunt/'
    elif parameters.hawk_dove:
        return 'populations/hawk_dove/'
    elif parameters.chicken:
        return 'populations/chicken/'

def makeDirectories(parameters):

    game_str = getGame(parameters)

    if parameters.a2c:
        base = 'populations/a2c/'
    elif parameters.pd and not parameters.pd_donation:
        base = game_str+'self'+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"
    elif (parameters.pd and parameters.pd_donation) and not parameters.sample_strategy and parameters.ppo:
        base = game_str+"ppo/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"
    elif (parameters.pd and parameters.pd_donation) and not parameters.sample_strategy and parameters.mean_team:
        base = game_str+"mean_team/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"
    elif (parameters.pd and parameters.pd_donation) and not parameters.sample_strategy and parameters.ninc > 0 and parameters.even_inc:
        base = game_str+"inc/even/"+str(parameters.ninc)+"-inc-agents/"+str(parameters.inc_deg)+"deg/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"
    elif (parameters.pd and parameters.pd_donation) and not parameters.sample_strategy and parameters.ninc > 0 and not parameters.even_inc:
        base = game_str+"inc/"+str(parameters.ninc)+"-inc-agents/"+str(parameters.inc_deg)+"deg/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"
    elif (parameters.pd and parameters.pd_donation) and not parameters.sample_strategy and parameters.attn:
        base = game_str+"attn/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"
    elif (parameters.pd and parameters.pd_donation) and not parameters.sample_strategy and parameters.per:
        base = game_str+"per/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"
    elif (parameters.pd and parameters.pd_donation) and not parameters.sample_strategy and parameters.beta_dist:
        base = game_str+"beta_"+str(parameters.beta_a)+"-"+str(parameters.beta_b)+"/HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"
    elif (parameters.pd and parameters.pd_donation) and not parameters.sample_strategy:
        base = game_str+"HOMOG/self"+str(parameters.selfW)+"team"+str(parameters.teamW)+"sys"+str(parameters.systemW)+"/"
    else:
        base = 'populations/'

    if not os.path.exists(base):
        os.makedirs(base)

def checkAsserts(parameters):
    assert not (parameters.beta_dist and parameters.per), "Cant be sampling with beta distirbution AND PER"
    assert (parameters.benefit > parameters.cost), "BENEFIT must be more than COST"
    assert parameters.benefit > (parameters.benefit - parameters.cost), "Temptation reward must be more than Cooperating"
    assert parameters.summaryLength < parameters.nepisodes, "summaryLength must be less than the number of episodes"
    assert parameters.endSummaryLength < parameters.nepisodes, "endSummaryLength must be less than the number of episodes"

    if parameters.pd_donation:
        assert (2*(parameters.benefit - parameters.cost)) > (parameters.benefit - parameters.cost), "C,C must have global optimal return"
    assert (parameters.selfW + parameters.teamW + parameters.systemW) == 1, "Reward preference weights must sum to 1, currently: selfW = {}, teamW = {}, systemW = {}".format(parameters.selfW, parameters.teamW, parameters.systemW)
