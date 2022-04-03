import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
#import tikzplotlib

def reward_plot(reward_cons_q, reward_cons_q_kappa):
    reward_mean = np.mean(reward_cons_q, axis=0)
    reward_std = np.std(reward_cons_q, axis=0)

    reward_min = reward_mean - reward_std
    reward_max = reward_mean + reward_std

    reward_mean_kappa = np.mean(reward_cons_q_kappa, axis=0)
    reward_std_kappa = np.std(reward_cons_q_kappa, axis=0)

    reward_min_kappa = reward_mean_kappa - reward_std_kappa
    reward_max_kappa = reward_mean_kappa + reward_std_kappa

    T = len(reward_mean)
    episode = np.arange(start=0, stop=T, step=10)
    episode2 = np.arange(start=0, stop=T, step=1)
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    '''
    lns1 = ax.plot(episode, reward_mean[episode], color='k', label='kappa=0', linewidth=1)
    ax.fill_between(episode2, reward_min[episode2], reward_max[episode2], color='k', alpha=0.3)
    #lns2 = ax.plot(episode, reward_mean_kappa[episode], color='b', label='kappa>0', linewidth=1)
    #ax.fill_between(episode2, reward_min_kappa[episode2], reward_max_kappa[episode2], color='b', alpha=0.3)
    '''
    lns1 = ax.plot(episode, reward_mean[episode], color='k', label='varphi=0.48', linewidth=1)
    ax.fill_between(episode2, reward_min[episode2], reward_max[episode2], color='k', alpha=0.3)
    lns2 = ax.plot(episode, reward_mean_kappa[episode], color='b', label='varphi=0.048', linewidth=1)
    ax.fill_between(episode2, reward_min_kappa[episode2], reward_max_kappa[episode2], color='b', alpha=0.3)

    ax.grid()
    ax.set_xlabel("Iteration t", fontsize='x-large')
    ax.set_ylabel("Total reward", fontsize='x-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right', fontsize='xx-large')
    plt.savefig('learning_reward.png', dpi=600)
    #tikzplotlib.save('learning_reward.tex')
    #plt.show()


def vio_plot(vio_cons_q1, vio_cons_q2, vio_cons_q1_kappa, vio_cons_q2_kappa):
    vio1_mean = np.mean(vio_cons_q1, axis=0)
    vio1_std = np.std(vio_cons_q1, axis=0)

    vio1_min = vio1_mean - vio1_std
    vio1_max = vio1_mean + vio1_std

    vio2_mean = np.mean(vio_cons_q2, axis=0)
    vio2_std = np.std(vio_cons_q2, axis=0)

    vio2_min = vio2_mean - vio2_std
    vio2_max = vio2_mean + vio2_std

    vio1_mean_kappa = np.mean(vio_cons_q1_kappa, axis=0)
    vio1_std_kappa = np.std(vio_cons_q1_kappa, axis=0)

    vio1_min_kappa = vio1_mean_kappa - vio1_std_kappa
    vio1_max_kappa = vio1_mean_kappa + vio1_std_kappa

    vio2_mean_kappa = np.mean(vio_cons_q2_kappa, axis=0)
    vio2_std_kappa = np.std(vio_cons_q2_kappa, axis=0)

    vio2_min_kappa = vio2_mean_kappa - vio2_std_kappa
    vio2_max_kappa = vio2_mean_kappa + vio2_std_kappa

    T = len(vio1_mean)
    episode = np.arange(start=0, stop=T, step=10)
    episode2 = np.arange(start=0, stop=T, step=1)
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)
    '''
    lns1 = ax.plot(episode, vio1_mean[episode], color='k', label='service kappa=0', linewidth=1)
    ax.fill_between(episode2, vio1_min[episode2], vio1_max[episode2], color='k', alpha=0.3)
    lns2 = ax.plot(episode, vio2_mean[episode], color='b', label='flow kappa=0', linewidth=1)
    ax.fill_between(episode2, vio2_min[episode2], vio2_max[episode2], color='b', alpha=0.3)

    #lns3 = ax.plot(episode, vio1_mean_kappa[episode], color='r', label='service kappa>0', linewidth=1)
    #ax.fill_between(episode2, vio1_min_kappa[episode2], vio1_max_kappa[episode2], color='r', alpha=0.3)
    #lns4 = ax.plot(episode, vio2_mean_kappa[episode], color='g', label='flow kappa>0', linewidth=1)
    #ax.fill_between(episode2, vio2_min_kappa[episode2], vio2_max_kappa[episode2], color='g', alpha=0.3)
    '''
    lns1 = ax.plot(episode, vio1_mean[episode], color='k', label='service varphi=0.48', linewidth=1)
    ax.fill_between(episode2, vio1_min[episode2], vio1_max[episode2], color='k', alpha=0.3)
    lns2 = ax.plot(episode, vio2_mean[episode], color='b', label='flow varphi=0.048', linewidth=1)
    ax.fill_between(episode2, vio2_min[episode2], vio2_max[episode2], color='b', alpha=0.3)

    lns3 = ax.plot(episode, vio1_mean_kappa[episode], color='r', label='service varphi=0.48', linewidth=1)
    ax.fill_between(episode2, vio1_min_kappa[episode2], vio1_max_kappa[episode2], color='r', alpha=0.3)
    lns4 = ax.plot(episode, vio2_mean_kappa[episode], color='g', label='flow varphi=0.048', linewidth=1)
    ax.fill_between(episode2, vio2_min_kappa[episode2], vio2_max_kappa[episode2], color='g', alpha=0.3)

    ax.grid()
    ax.set_xlabel("Iteration t", fontsize='x-large')
    ax.set_ylabel("Constraint Value", fontsize='medium')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.ylim(-0.05, 0.05)
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='lower right', fontsize='xx-large')
    plt.tight_layout()
    plt.savefig('learning_violation.png', dpi=600)
    #tikzplotlib.save('learning_violation.tex')
    # plt.show()



if __name__ == '__main__':
    reward_cons_q = np.load('reward.npy')
    vio_cons_q1 = np.load('service.npy')
    vio_cons_q2 = np.load('flow.npy')
    reward_cons_q_kappa = np.load('reward0048.npy')
    vio_cons_q1_kappa = np.load('service0048.npy')
    vio_cons_q2_kappa = np.load('flow0048.npy')
    reward_plot(reward_cons_q, reward_cons_q_kappa)
    vio_plot(vio_cons_q1, vio_cons_q2, vio_cons_q1_kappa, vio_cons_q2_kappa)
