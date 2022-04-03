import numpy as np
import sys
import time


class environment:
    def __init__(self, NUM_STATE, NUM_ACTION_1, NUM_ACTION_2, min_action_1, max_action_1, min_action_2, max_action_2, gamma, rho):
        self.state = np.arange(NUM_STATE)
        self.NUM_STATE = NUM_STATE

        self.action_service = np.linspace(min_action_1, max_action_1, NUM_ACTION_1)
        self.action_flow = np.linspace(min_action_2, max_action_2, NUM_ACTION_2)
        self.NUM_ACTION_1 = NUM_ACTION_1
        self.NUM_ACTION_2 = NUM_ACTION_2

        self.transition_tensor = self.initial_transition()

        self.reward = self.initial_reward()
        self.service_constraint = self.initial_service_constraint()
        self.flow_constraint = self.initial_flow_constraint()

        self.gamma = gamma

        self.current_state = self.sample_initial_state(rho)
        self.internal_step_count = 0

    def sample_initial_state(self, distribution):
        count = 0
        prob = np.random.rand()
        for state in range(self.NUM_STATE):
            if prob > count and prob <= count + distribution[state]:
                self.current_state = state
                return self.current_state
            count = count + distribution[state]

    def initial_transition(self):
        transition_tensor = np.zeros([self.NUM_STATE, self.NUM_ACTION_1, self.NUM_ACTION_2, self.NUM_STATE])
        for state in range(self.NUM_STATE):
            for service in range(self.NUM_ACTION_1):
                for flow in range(self.NUM_ACTION_2):
                    transition_tensor[state, service, flow, :] = self.transition_pro(state, service, flow)
        return transition_tensor

    def transition_pro(self, current_state, service_action, flow_action):
        next_state_pro = np.zeros(self.NUM_STATE)
        service_pro = self.action_service[service_action]
        flow_pro = self.action_flow[flow_action]
        if current_state >= 1 and current_state < self.NUM_STATE - 1:
            next_state_pro[current_state - 1] = service_pro * (1 - flow_pro)
            next_state_pro[current_state + 1] = flow_pro * (1 - service_pro)
            next_state_pro[current_state] = 1 - next_state_pro[current_state - 1] - next_state_pro[current_state + 1]
        if current_state == self.NUM_STATE - 1:
            next_state_pro[current_state - 1] = service_pro
            next_state_pro[current_state] = 1 - service_pro
        if current_state == 0:
            next_state_pro[current_state + 1] = flow_pro * (1 - service_pro)
            next_state_pro[current_state] = 1 - flow_pro * (1 - service_pro)
        return next_state_pro


    def get_next_state(self, service_action, flow_action):
        next_state_pro = self.transition_pro(self.current_state, service_action, flow_action)
        np.random.seed()
        temp = np.random.rand()
        pro = 0
        for i in range(NUM_STATE):
            if temp > pro and temp<= pro + next_state_pro[i]:
                return i
            else:
                pro += next_state_pro[i]
        return 'Error'

    def initial_reward(self):
        R = np.zeros([self.NUM_STATE, self.NUM_ACTION_1, self.NUM_ACTION_2])
        for state in range(self.NUM_STATE):
            R[state, :, :] = -state + 5
        return R

    def initial_service_constraint(self):
        C1 = np.zeros([self.NUM_STATE, self.NUM_ACTION_1, self.NUM_ACTION_2])
        for service_action in range(self.NUM_ACTION_1):
            C1[:, service_action, :] = -10 * self.action_service[service_action] + 3
        return C1

    def initial_flow_constraint(self):
        C2 = np.zeros([self.NUM_STATE, self.NUM_ACTION_1, self.NUM_ACTION_2])
        for flow_action in range(self.NUM_ACTION_2):
            C2[:, :, flow_action] = -8 * ((self.action_flow[flow_action] - 1) ** 2) + 1.2
        return C2

    def env_reset(self, rho):
        self.current_state = self.sample_initial_state(rho)
        self.internal_step_count = 0
        return self.current_state

    def env_step(self, action_service, action_flow, output=0):
        current_reward = self.reward[self.current_state, action_service, action_flow]
        current_service_constraint = self.service_constraint[self.current_state, action_service, action_flow]
        current_flow_constraint = self.flow_constraint[self.current_state, action_service, action_flow]
        next_state = self.get_next_state(action_service, action_flow)
        if self.current_state == 'Error':
            print('Error')
            return -1
        if output == 1:
            out_str = '\r ' + "Step:" + str(self.internal_step_count) + " Reward:" + str(current_reward) \
                      + " Constraint_service:" + str(current_service_constraint) + " Constraint_flow:" + str(current_flow_constraint)\
                      + " From state:" + str(self.current_state) + " To state:" + str(next_state) + '\n'
            sys.stdout.write(out_str)
            time.sleep(0.1)
        self.current_state = next_state
        self.internal_step_count += 1
        return self.current_state, current_reward, current_service_constraint, current_flow_constraint

    def test(self, epoch, rho):
        self.env_reset(rho)
        for i in range(epoch):
            rnd_action1 = np.random.randint(low=0, high=NUM_ACTION_1)
            rnd_action2 = np.random.randint(low=0, high=NUM_ACTION_2)
            self.env_step(rnd_action1, rnd_action2, 1)
        return 0

NUM_STATE = 5
NUM_ACTION_1 = 4
NUM_ACTION_2 = 4
min_action_1 = 0.2
max_action_1 = 0.8
min_action_2 = 0.4
max_action_2 = 0.7
gamma = 0.5
rho = np.ones(NUM_STATE) / NUM_STATE

my_env = environment(NUM_STATE, NUM_ACTION_1, NUM_ACTION_2, min_action_1, max_action_1, min_action_2, max_action_2, gamma, rho)
#print(my_env.reward.reshape(NUM_STATE, NUM_ACTION_1*NUM_ACTION_2))
#my_env.test(100, rho)

