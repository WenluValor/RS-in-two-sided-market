import math
import numpy as np
import matplotlib.pyplot as plt

def set_param(T_value, mu_value, c_value):
    global m, T, mu, c, eta, X, y, I, vec_p
    m = mu_value.shape[0]
    T = T_value
    mu = mu_value
    c = c_value
    eta = math.sqrt(8 * math.log(m) / T)

    X = np.zeros([T]) - 1
    y = np.zeros([T]) - 1
    I = np.zeros([T]) - 1
    vec_p = np.zeros([m]) + 1 / m
    return

def has_incentive(t):
    hist_X = X[0: t]
    hist_y = y[0: t]
    hist_reward = sum(hist_X * hist_y) / sum(hist_y)

    if hist_reward >= c[t]:
        return True
    else:
        return False

def set_reward(item_ind: np.array, start: int):
    length = item_ind.shape[0]
    tmp = np.zeros([length])
    for i in range(length):
        tmp[i] = mu[int(item_ind[i])]
    X[start: start + length] = tmp + np.random.uniform(-0.1, 0.1, size=length)

def get_recommendation(vec_p):
    U = np.random.uniform(0, 1)
    prev_sum = 0
    if U < vec_p[0]:
        return 0
    for j in range(vec_p.shape[0] - 1):
        prev_sum += vec_p[j]
        if (U >= prev_sum) & (U < prev_sum + vec_p[j + 1]):
            return j + 1

def updata_p(L: np.array):
    denom = sum(np.exp(-eta * L))
    for i in range(m):
        vec_p[i] = math.exp(-eta * L[i]) / denom
    return

def initialize(t=0):
    reco = get_recommendation(vec_p=vec_p)
    I[t] = reco
    y[t] = 1
    set_reward(item_ind=np.array(reco).reshape(1), start=t)
    t += 1
    return t

def explore(t):
    start = t
    L = np.zeros([m])
    for t in range(start, T):
        est_loss = get_est_loss(t=t)
        L += est_loss
        updata_p(L=L)

        reco = get_recommendation(vec_p=vec_p)
        if has_incentive(t=t):
            I[t] = reco
            y[t] = 1
            set_reward(item_ind=np.array(reco).reshape(1), start=t)
        else:
            y[t] = 0
    return

def get_est_loss(t):
    est_loss = np.zeros([m])
    i = int(I[t - 1])
    est_loss[i] = -X[t - 1] * y[t - 1] / vec_p[i]
    return est_loss

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
    plt.show()
    return

if __name__ == '__main__':
    T = 10000
    mu = np.array([0.5, 0.3, 0.8])
    c_value = np.random.uniform(0, 1, size=T)
    set_param(T_value=T, mu_value=mu, c_value=c_value)
    t = initialize()
    stop = explore(t=t)
    draw_plot()
    exit(0)