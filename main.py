'''
Author:         David Radke
Affiliation:    University of Waterloo
Date:           May 2, 2022
'''

import tensorflow as tf
import numpy as np
import copy
import os
import logging
import matplotlib.pyplot as plt
tf.get_logger().setLevel(logging.ERROR)
from multiprocessing import Pool
import scipy
import time

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as mtri

import sys
from parser import process_args

import random
import environment
import agency
import utils
import plot_policy_mtx
import plots


class Defaults:
    batch_size = 32 #4 #How many experience traces to use for each training step.
    nteams = 2
    nactions = 2
    nagents = 2
    target_update_period = 10
    replay_buffer_max_length = 100000
    learning_rate = 0.0001
    nepisodes = 50000
    gamma = 0.99
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.05
    copy_step = 25
    summaryLength = 2000

def playPrisonersDilemma(epi, parameters, population_obj, env):
    for play in range(parameters.plays_per_episode):
        list_of_agents = []
        for t in population_obj._teams.keys():
            list_of_agents+=population_obj._teams[t]._team
            population_obj._teams[t]._team_reward[epi] = 0
            population_obj._teams[t]._team_games[epi] = 0

        exps = {}
        r_population = 0
        ''' play the PD, record experiences '''
        for agent_idx, agent in enumerate(list_of_agents):
            opponent = env.sample(population_obj, agent)
            if parameters.mean_team:
                action1, agent_q_vals = agent._trainNet.get_action(opponent._team, env._epsilon, team_q=population_obj._teams[agent._team_num]._team_q[opponent._team_num])
                action2, opp_q_vals = opponent._trainNet.get_action(agent._team, env._epsilon)
            else:
                action1, agent_q_vals = agent._trainNet.get_action(opponent._team, env._epsilon)
                action2, opp_q_vals = opponent._trainNet.get_action(agent._team, env._epsilon)

            r_agent, r_opponent, done = env.step(action1, action2, agent, opponent, play)


            ''' records mutual cooperation, sucker, temptation, and mutual defection for each agent '''
            agent.recordResult(action1, action2, epi, opponent, is_focal=True)
            opponent.recordResult(action2, action1, epi, agent)

            ''' save agent q values '''
            agent._q_vals[opponent._team_num, :,epi,play] = agent_q_vals
            # if epi%parameters.summaryLength == 0 and agent_idx == 0 and play == 0:
            #     print("AGENT 0-0 Q VALUES: ", agent_q_vals)

            ''' update agent's team total reward '''
            population_obj._teams[agent._team_num]._team_reward[epi] += r_agent
            population_obj._teams[agent._team_num]._team_games[epi] += 1

            ''' update opponent's team total reward '''
            population_obj._teams[opponent._team_num]._team_reward[epi] += r_opponent
            population_obj._teams[opponent._team_num]._team_games[epi] += 1

            r_population+=(r_agent+r_opponent)
            exps[agent._id] = {'opp': opponent, 's_opp': agent._team, 'a_opp': action2, 'r_opp': r_opponent, 's2_opp': agent._team,
                                's_agent': opponent._team, 'a_agent': action1, 'r_agent': r_agent, 's2_agent': opponent._team, 'done': done}

        ''' calculate credo reward and update '''
        for agent_idx, agent in enumerate(list_of_agents):
            opponent = exps[agent._id]['opp']
            w_reward_agent = ((agent._selfW * exps[agent._id]['r_agent']) +
                        ((agent._teamW * population_obj._teams[agent._team_num]._team_reward[epi]) / population_obj._teams[agent._team_num]._team_games[epi]) +
                        ((agent._systemW * r_population) / (2*parameters.nagents*parameters.nteams)))

            w_reward_opp = ((opponent._selfW * exps[agent._id]['r_opp']) +
                        ((opponent._teamW * population_obj._teams[opponent._team_num]._team_reward[epi]) / population_obj._teams[opponent._team_num]._team_games[epi]) +
                        ((opponent._systemW * r_population) / (2*parameters.nagents*parameters.nteams)))

            agent._rewards+=w_reward_agent
            agent._reward_lst[epi,play] = w_reward_agent
            agent._action_history[opponent._team_num].append(exps[agent._id]['a_agent'])
            agent._action_mtx[opponent._team_num,epi,play] = exps[agent._id]['a_agent']

            agent_exp = {'s': opponent._team, 'a': exps[agent._id]['a_agent'], 'r': w_reward_agent, 's2': opponent._team, 'done': done} #opponent._team
            opp_exp = {'s': agent._team, 'a': exps[agent._id]['a_opp'], 'r': w_reward_opp, 's2': agent._team, 'done': done}

            agent._trainNet.add_experience(agent_exp)
            opponent._trainNet.add_experience(opp_exp)

            if parameters.a2c:
                a_loss, c_loss = agent._trainNet.train()
            else:
                agent._iter+=1
                loss = agent._trainNet.train(agent._targetNet, agent)
                if isinstance(loss, int):
                    agent._losses.append(loss)
                else:
                    agent._losses.append(loss.numpy())

                if agent._iter % agent._copy_step == 0:
                    agent._targetNet.copy_weights(agent._trainNet)

        for agent_idx, agent in enumerate(list_of_agents):
            agent._len_buffer.append(len(agent._trainNet.experience['s']))
    return



def playEpisodes(parameters, teams, agent_lst, env, population_obj):

    for epi in range(parameters.nepisodes):
        env._epsilon = max(env._min_epsilon, env._epsilon*parameters.decay)
        playPrisonersDilemma(epi, parameters, population_obj, env)

        if epi%parameters.summaryLength == 0 and epi > 0:
            population_obj.printSummary(epi, env)
            population_obj.logSummary(epi, env)


    makeGraphs(population_obj, parameters)




if __name__ == "__main__":
    start_time = time.time()

    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    parameters.endSummaryLength = int(parameters.nepisodes*0.25)

    if parameters.pd_donation:
        parameters.pd = True

    utils.checkAsserts(parameters)

    if parameters.sample_strategy:
        parameters.pd_donation = True


    # random checks for side projects I implemented
    if parameters.cd:
        parameters.nactions = parameters.nteams + (parameters.nteams-2)
    elif parameters.ohc:
        parameters.nactions = parameters.nteams + (parameters.nteams-1)
    elif (parameters.nteams > parameters.nactions) and not (parameters.pd or parameters.pd_donation or parameters.stag_hunt or parameters.hawk_dove or parameters.chicken):
        parameters.nactions = parameters.nteams

    if parameters.pd:
        parameters.nactions = 2

    utils.makeDirectories(parameters)

    print(parameters)

    # if not parameters.sample_strategy:
    seed_rand = parameters.seed #random.randint(0,100)
    # seed_rand = random.randint(0,100)
    from numpy.random import seed
    seed(seed_rand)
    tf.random.set_seed(seed_rand)
    print("Seed: ", seed_rand)

    agent_lst, teams, env, population_obj = agency.makePopulation(parameters)
    playEpisodes(parameters, teams, agent_lst, env, population_obj)

    runtime_seconds = time.time() - start_time
    print("************** Program runtime **************")
    print("In seconds: ", runtime_seconds)
    print("In minutes: ", runtime_seconds / 60)
    print("In hours:   ", (runtime_seconds / 60) / 60)
    exit()
