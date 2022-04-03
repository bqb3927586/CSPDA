import numpy as np
from environment_queue import environment
import matplotlib.pyplot as plt
import multiprocessing
from tqdm.contrib.concurrent import process_map

NUM_STATE = 5
NUM_ACTION_1 = 4
NUM_ACTION_2 = 4
min_action_1 = 0.2
max_action_1 = 0.8
min_action_2 = 0.4
max_action_2 = 0.7
gamma = 0.5
rho = np.ones(NUM_STATE) / NUM_STATE

varphi = 0.048
T = 100000
I = 2
alpha = np.sqrt(NUM_STATE) / (1 - gamma) / varphi / np.sqrt(T * I)
#alpha = 0.01
beta = (1 - gamma) * varphi * np.sqrt(
    np.log(NUM_STATE * NUM_ACTION_1 * NUM_ACTION_2) / T / NUM_STATE / NUM_ACTION_1 / NUM_ACTION_2)

#beta = 0.01


M = 4 * (1 / varphi + 1 / (1 - gamma) + 2 / (1 - gamma) / varphi)
# M = 10
c = 0.02
kappa = 2 * c / (1 - gamma) * np.sqrt(
    I * NUM_STATE * NUM_ACTION_1 * NUM_ACTION_2 * np.log(NUM_STATE * NUM_ACTION_1 * NUM_ACTION_2) / T)
kappa = 0
delta = 0.1
test_num = 100
bound_u = 4 / varphi
def sample_state_action(my_env, distribution):
    count = 0
    prob = np.random.rand()
    for state in range(my_env.NUM_STATE):
        for service in range(my_env.NUM_ACTION_1):
            for flow in range(my_env.NUM_ACTION_2):
                if prob > count and prob <= count + distribution[state, service, flow]:
                    return state, service, flow
                else:
                    count = count + distribution[state, service, flow]
    print(prob, count, 'ERROR')
    print(distribution)
    return 0

'''
def sample_initial_state(my_env, distribution):
    count = 0
    prob = np.random.rand()
    for state in range(my_env.NUM_STATE):
        if prob > count and prob <= count + distribution[state]:
            return state
        count = count + distribution[state]
'''

def CSPDA(my_env):
    u1 = 0
    u2 = 0
    v = 0
    uniform = np.ones([my_env.NUM_STATE, my_env.NUM_ACTION_1, my_env.NUM_ACTION_2]) / (my_env.NUM_STATE * my_env.NUM_ACTION_1 * my_env.NUM_ACTION_2)
    Lambda = np.copy(uniform)
    output_Lambda = np.copy(uniform)
    total_reward = np.zeros(T)
    vio_service = np.zeros(T)
    vio_flow = np.zeros(T)
    for t in range(T):
        #print('Lambda', Lambda)
        zeta = (1 - delta) * np.copy(Lambda) + delta * uniform
        #print(np.sum(zeta))
        state, service, flow = sample_state_action(my_env, zeta)
        s0 = my_env.sample_initial_state(rho)
        my_env.current_state = state
        ss = my_env.get_next_state(service, flow)
        r_sa = my_env.reward[state, service, flow]
        g1_sa = my_env.service_constraint[state, service, flow]
        g2_sa = my_env.flow_constraint[state, service, flow]
        nabla_u1 = Lambda[state, service, flow] * g1_sa / zeta[state, service, flow] - kappa
        nabla_u2 = Lambda[state, service, flow] * g2_sa / zeta[state, service, flow] - kappa
        u1 = u1 - alpha * nabla_u1
        u2 = u2 - alpha * nabla_u2
        #print(u1, u2)
        '''
        if u1<0 and u2<0:
            u1=0
            u2=0
        elif u1<0:
            u1=0
            if u2>bound_u:
                u2=bound_u
        elif u2<0:
            u2=0
            if u1 > bound_u:
                u1=bound_u
        elif np.abs(u1)+np.abs(u2)>bound_u:
            u1 = 0.5 * (bound_u + u1 - u2)
            u2 = 0.5 * (bound_u + u2 - u1)
        '''
        #print(u1, u2,'\n')
        #if u1 < 0:
        #    u1 = 0
        #if u2 < 0:
        #    u2 = 0
        #if np.abs(u1) + np.abs(u2) > bound_u:
        #    u1 = 0.5 * (bound_u + u1 - u2)
        #    u2 = 0.5 * (bound_u + u2 - u1)
        es0 = np.zeros(my_env.NUM_STATE)
        es = np.zeros(my_env.NUM_STATE)
        ess = np.zeros(my_env.NUM_STATE)
        es0[s0] = 1 - gamma
        es[state] = 1
        ess[ss] = 1
        nabla_v = es0 + Lambda[state, service, flow] * (gamma * ess - es) / zeta[state, service, flow]
        v = v - alpha * nabla_v
        #for s in range(my_env.NUM_STATE):
        #    v[s] = np.max([2 * (1 / (1 - gamma) + 2 / (1 - gamma) / varphi), v[s]])
        Z_sa = r_sa + gamma * v[ss] - v[state] + u1 * g1_sa + u2 * g2_sa
        nabla_lambda = (Z_sa - M) / zeta[state, service, flow]
        Lambda[state, service, flow] = Lambda[state, service, flow] * np.exp(beta * nabla_lambda)
        Lambda = Lambda / np.sum(Lambda)
        output_Lambda = (t / (t + 1)) * output_Lambda + (1 / (t + 1)) * Lambda
        total_reward[t] = np.sum(np.multiply(output_Lambda, my_env.reward))
        vio_service[t] = np.sum(np.multiply(output_Lambda, my_env.service_constraint))
        vio_flow[t] = np.sum(np.multiply(output_Lambda, my_env.flow_constraint))
    return total_reward, vio_service, vio_flow


def fig_plot(total_reward, vio_service, vio_flow, T):
    iteration = np.arange(T)
    plt.plot(iteration, total_reward, color='r', label='Objective', linewidth=1)
    plt.plot(iteration, vio_service, color='g', label='Service', linewidth=1)
    plt.plot(iteration, vio_flow, color='b', label='Flow', linewidth=1)
    plt.grid()
    plt.legend(loc=0)
    plt.savefig('learning.png', dpi=600)
    plt.show()


if __name__ == "__main__":
    my_env = []
    for i in range(test_num):
        my_env.append(environment(NUM_STATE, NUM_ACTION_1, NUM_ACTION_2, min_action_1, max_action_1, min_action_2, max_action_2,
                         gamma, rho))

    result_reward = np.zeros([test_num, T])
    result_vio1 = np.zeros([test_num, T])
    result_vio2 = np.zeros([test_num, T])
    print('Start Parallelization')
    pool = multiprocessing.Pool()
    print('Parallel Learning')
    result = process_map(CSPDA, my_env)
    print('Unpack Result')
    for i in range(test_num):
        result_reward[i, :] = result[i][0]
        result_vio1[i, :] = result[i][1]
        result_vio2[i, :] = result[i][2]
    np.save('reward0048.npy', result_reward)
    np.save('service0048.npy', result_vio1)
    np.save('flow0048.npy', result_vio2)
    #total_reward, vio_service, vio_flow = CSPDA(T, rho, gamma, alpha, beta, varphi, M, kappa, delta, my_env)
    #fig_plot(total_reward, vio_service, vio_flow, T)
