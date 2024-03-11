import math
import numpy as np
import matplotlib.pyplot as plt

def set_param(T_value, mu_value, c_star_value):
    global m, T, mu, c_star, X, y, I, chosen_count, est_X
    m = mu_value.shape[0]
    T = T_value
    mu = mu_value
    c_star = c_star_value

    X = np.zeros([T]) - 1
    y = np.zeros([T]) - 1
    I = np.zeros([T]) - 1
    chosen_count = np.zeros([m])
    est_X = np.zeros([m])
    return

def initialize(t=0):
    for i in range(m):
        I[t] = i
        y[t] = 1
        set_reward(item_ind=np.array(i).reshape(1), start=t)

        chosen_count[i] += 1
        est_X[i] = ((chosen_count[i] - 1) * est_X[i] + X[t]) / (chosen_count[i])

        t += 1
    return t

def explore(t):
    start = t
    flag = True
    stop = -1
    for t in range(start, T):
        reco = get_arm()
        if has_incentive(t=t):
            I[t] = reco
            y[t] = 1
            set_reward(item_ind=np.array(reco).reshape(1), start=t)

            i = int(reco)
            chosen_count[i] += 1
            est_X[i] = ((chosen_count[i] - 1) * est_X[i] + X[t]) / (chosen_count[i])
        else:
            y[t] = 0
            if flag:
                stop = t
            flag = False
    return stop

def get_arm():
    bound = est_X + np.sqrt(2 * math.log(sum(chosen_count)) / chosen_count)
    item = np.argmax(bound)
    return item

def has_incentive(t):
    hist_X = X[0: t]
    hist_y = y[0: t]
    hist_reward = sum(hist_X * hist_y) / sum(hist_y)

    if hist_reward >= c_star:
        return True
    else:
        return False

def set_reward(item_ind: np.array, start: int):
    length = item_ind.shape[0]
    tmp = np.zeros([length])
    for i in range(length):
        tmp[i] = mu[int(item_ind[i])]
    X[start: start + length] = tmp + np.random.uniform(-0.1, 0.1, size=length)

def R(end: int):
    I_list = I[0: end]
    y_list = y[0: end]
    sum1 = get_l(I_list=I_list, y_list=y_list)
    sum2 = np.zeros([m])

    y_list = np.ones([end])
    for i in range(m):
        item_list = np.zeros([end]) + i
        sum2[i] = get_l(I_list=item_list, y_list=y_list)
    return sum1 - min(sum2)


def get_l(I_list: np.array, y_list: np.array):
    sum = 0
    mat_l = get_mat_l()
    for i in range(I_list.shape[0]):
        i1 = int(I_list[i])
        i2 = int(y_list[i])
        sum += mat_l[i1, i2]
    return sum

def get_mat_l():
    mat_l = np.zeros([m, 2])
    for i in range(m):
        for j in range(2):
            mat_l[i, j] = max(mu - mu[i] * j)
    return mat_l

def draw_plot():
    size = 10
    x = np.arange(0, T, step=int(T / size))
    y_points = np.zeros([size])
    for i in range(size):
        y_points[i] = R(end=int(T / size) * i)
    plt.plot(x, y_points)
    # plt.plot(x, y_points, color='blue', label='regret')
    # plt.legend(loc='upper right')
    plt.show()
    return

if __name__ == '__main__':
    np.random.seed(9)
    c_star = 0.4
    mu = np.array([0.5, 0.3, 0.8])
    T = 1000
    set_param(T_value=T, mu_value=mu, c_star_value=c_star)
    t = initialize()
    stop = explore(t=t)
    print(stop)
    draw_plot()


    '''
    y[0: 10] = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    I[0: 10] = np.array([0, 2, 1, 0, -1, -1, -1, -1, -1, -1])
    y_points = R(end=10)
    p_pints = principal_l(end=10)
    print(y_points)
    print(p_pints)
    '''



    exit(0)