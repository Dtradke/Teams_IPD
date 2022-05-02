'''
Author:         David Radke
Affiliation:    University of Waterloo
Date:           May 2, 2022
'''

import argparse

def process_args(args, defaults):
    """Handle the command line and return an object containing all the parameters.
    Arguments:
        args     - list of command line arguments (not including executable name)
        defaults - a name space with variables corresponding to each of the required default command line values.
    """

    parser = argparse.ArgumentParser(description='Tiered Cooperation')
    parser.add_argument('--env', default='cooperation', help='environment')
    parser.add_argument('-o', '--output', default='./Center/', help='Directory to save data to')
    parser.add_argument('--seed', default=39, type=int, help='Random seed')
    parser.add_argument('--nteams', default=5, type=int, help='Number of Teams.')
    parser.add_argument('--nactions', default=2, type=int, help='Number of Actions.')
    parser.add_argument('--nagents', default=5, type=int, help='Number of Agents per-team.')
    parser.add_argument('--ninc', default=0, type=int, help='Number of incompetent agents.')
    parser.add_argument('--inc_deg', default=0.5, type=float, help='Degree of incompetence. 0 = fully rational, 1 = fully incompetent.')
    parser.add_argument('--even_inc', default=False, type=bool, help='Fills up the teams evenly with rogue agents (not all on team 0 first, but each gets 1 before any get 2).')


    parser.add_argument('--target_update_period', default=10, type=int, help='When to update Target.')
    parser.add_argument('--replay_buffer_max_length', default=100000, type=int, help='Replay buffer max length.')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--nepisodes', default=100000, type=int, help='number of episodes to run') #was 10000
    parser.add_argument('--plays_per_episode', default=1, type=int, help='number of game plays per episode to run') # try changing this to 200
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--epsilon', default=0.99, type=float, help='Epsilon')
    parser.add_argument('--decay', default=0.9999, type=float, help='Epsilon decay')
    parser.add_argument('--min_epsilon', default=0.05, type=float, help='Discount factor')
    parser.add_argument('--copy_step', default=2000, type=int, help='When to copy network') # was 25
    parser.add_argument('--summaryLength', default=2000, type=int, help='Number of episodes to save for analysis')
    parser.add_argument('--endSummaryLength', default=25000, type=int, help='Number of episodes to save for analysis')
    parser.add_argument('--cd', default=False, type=bool, help='Coordination mtx is a distribution')
    parser.add_argument('--ohc', default=False, type=bool, help='Coordination mtx is one hot for all, no distribution')
    parser.add_argument('--pd', default=False, type=bool, help='Prisoners Dilemma?')
    parser.add_argument('--pd_donation', default=False, type=bool, help='Prisoners Dilemma donation game?')
    parser.add_argument('--stag_hunt', default=False, type=bool, help='Stag hunt game?')
    parser.add_argument('--hawk_dove', default=False, type=bool, help='Hawk dove game?')
    parser.add_argument('--chicken', default=False, type=bool, help='Chicken game?')


    # parser.add_argument('--cost', default=1.0, type=float, help='Cost of donation game')
    # parser.add_argument('--benefit', default=5.0, type=float, help='Benefit of donation game')
    parser.add_argument('--cost', default=1, type=int, help='Cost of donation game')
    parser.add_argument('--benefit', default=5, type=int, help='Benefit of donation game')
    parser.add_argument('--save_eig_hist', default=10000, type=int, help='when to save eigenvalue histogram.')
    parser.add_argument('--save_exp', default=10000, type=int, help='when to save experience replay.')

    parser.add_argument('--team_sample_weight', default=1.0, type=float, help='How often the team of an agent is sampled')
    parser.add_argument('--mean_team', default=False, type=bool, help='Play the mean Q val of team')
    parser.add_argument('--attn', default=False, type=bool, help='Play the attention Q val of team')

    # how to replace buffer
    parser.add_argument('--replace_random', default=False, type=bool, help='Replace replay buffer experiences with random')
    parser.add_argument('--replace_similar', default=False, type=bool, help='Replace replay buffer experiences with most similar experience')

    # sampling
    parser.add_argument('--per', default=False, type=bool, help='Prioritized Experience Replay?')
    parser.add_argument('--beta_dist', default=False, type=bool, help='Sample from Experience Replay using Beta Distribution?')
    parser.add_argument('--beta_a', default=2.5, type=int, help='Value of a for beta sampling')
    parser.add_argument('--beta_b', default=1, type=int, help='Value of b for beta sampling')

    parser.add_argument('--sample_strategy', default=False, type=bool, help='Do teams sample strategy from fixed set?')
    parser.add_argument('--selfSS', default=3, type=int, help='Amount of self focused teams')
    parser.add_argument('--teamSS', default=1, type=int, help='Amount of team focused teams')
    parser.add_argument('--systemSS', default=1, type=int, help='Amount of system focused teams')
    parser.add_argument('--selfW', default=1.0, type=float, help='Reward weight for self')
    parser.add_argument('--teamW', default=0.0, type=float, help='Reward weight for team')
    parser.add_argument('--systemW', default=0.0, type=float, help='Reward weight for system')
    parser.add_argument('--sigma', default=1, type=float, help='Sigma for the coordistribution. Mean of gaussian distribution.')
    parser.add_argument('--load_model', default=False, type=bool, help='Do we load old models?')
    parser.add_argument('--load_agents', default=False, type=bool, help='Load previous agents?')
    parser.add_argument('--make_graphs', default=False, type=bool, help='Only make graphs given previous agents?')
    parser.add_argument('--batch_size', default=32, type=int, help='Minibatch size')
    parser.add_argument('--a2c', default=False, type=bool, help='is A2C?')
    parser.add_argument('--ppo', default=False, type=bool, help='is PPO?')

    parser.add_argument('--hex', default=False, type=bool, help='Make hexagonal graphs?')
    parser.add_argument('--check_hex', default=False, type=bool, help='Enough data for hex graph?')
    parser.add_argument('--check_buffer', default=False, type=bool, help='Calculates buffer stats for a previous run.')
    parser.add_argument('--atp', default=False, type=bool, help='Analyze team play?')
    parser.add_argument('--aqv', default=False, type=bool, help='Analyze q values?')
    parser.add_argument('--storytime', default=False, type=bool, help='Tell the story of defection')
    parser.add_argument('--gif', default=False, type=bool, help='Make GIF of defection')

    parameters = parser.parse_args(args)

    return parameters

if __name__ == '__main__':
    pass
