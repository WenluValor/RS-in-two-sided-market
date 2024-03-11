import math
import numpy as np
import matplotlib.pyplot as plt


def set_param(lmd, tau, T_value, mu_value, c_star_value):
    global theta, m, T, mu, c_star, k, X, y, I, mu_hat, L
    m = mu_value.shape[0]
    T = T_value
    mu = mu_value
    c_star = c_star_value

    theta = get_theta(tau=tau)
    k = int(get_k(theta=theta, lmd=lmd))
    k = 1000

    X = np.zeros([T]) - 1
    y = np.zeros([T]) - 1
    I = np.zeros([T]) - 1
    mu_hat = np.zeros([m]) - 1
    L = np.zeros([m])

    L[0] = k
    return

def get_theta(tau):
    return 4 * m**2 / (tau * (1 - c_star - tau))

def get_k(theta, lmd):
    A = 9 / (2 * lmd ** 2) * math.log(20 * m / lmd)
    B = theta ** 2 * math.log(T * theta)
    return max(A, B)

def set_reward(item_ind: np.array, start: int):
    length = item_ind.shape[0]
    tmp = np.zeros([length])
    for i in range(length):
        tmp[i] = mu[int(item_ind[i])]
    X[start: start + length] = tmp + np.random.uniform(-0.1, 0.1, size=length)

def update_a(item: int):
    return np.argmax(mu_hat[0: item])

def update_p(item: int):
    vec_p = np.zeros([m])
    a = update_a(item=item)
    vec_p[item] = get_p(item=item)
    vec_p[a] = 1 - vec_p[item]
    return vec_p

def get_p(item: int):
    M_hat = get_M_hat(item=item)
    return lmd / (2 * (c_star - M_hat) + lmd) * (M_hat < c_star) + (M_hat >= c_star)

def get_M_hat(item: int):
    num = int(sum(L[0: item]))
    tmp = X * y
    return sum(tmp[0: num]) / num

def get_recommendation(vec_p):
    U = np.random.uniform(0, 1)
    prev_sum = 0
    if U < vec_p[0]:
        return 0
    for j in range(vec_p.shape[0] - 1):
        prev_sum += vec_p[j]
        if (U >= prev_sum) & (U < prev_sum + vec_p[j + 1]):
            return j + 1

def incentive(t=0):
    I[0: k] = 0
    y[0: k] = 1
    set_reward(item_ind=np.zeros([k]), start=0)
    mu_hat[0] = np.mean(X[0: k])
    t = k

    for arm in range(1, m):
        vec_p = update_p(item=arm)
        count = 0
        sum = 0
        while count < k:
            reco = get_recommendation(vec_p=vec_p)
            I[t] = reco
            y[t] = 1
            set_reward(item_ind=np.array(reco).reshape(1), start=t)

            if reco != 0:
                sum += X[t]
                count += 1

            t += 1
            L[arm] += 1
        mu_hat[arm] = sum / k
    return t

def explore(t: int):
    arm_list = np.arange(m)
    q = k
    while arm_list.shape[0] > 1:
        cur_max = max(mu_hat)
        bound = max(cur_max, c_star)
        gap = math.sqrt(math.log(T * theta) / (2 * q))
        tmp_list = []
        for i in range(arm_list.shape[0]):
            if mu_hat[int(arm_list[i])] + gap >= bound:
                tmp_list.append(int(arm_list[i]))
        arm_list = np.array(tmp_list)

        for i in range(arm_list.shape[0]):
            I[t] = arm_list[i]
            y[t] = 1
            set_reward(item_ind=np.array(arm_list[i]).reshape(1), start=t)
            mu_hat[int(I[t])] = (mu_hat[int(I[t])] * q + X[t]) / (q + 1)
            t += 1

        q += 1
    return [arm_list[0], t]

def exploit(arm: int, t: int):
    start = t
    item_ind = arm * np.ones([T - start])
    I[t: T] = arm
    y[t: T] = 1
    set_reward(item_ind=item_ind, start=t)

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
        if i2 == 0:
            i1 = 0
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    c_star = 0.4
    tau = 0.2
    mu = np.array([0.5, 0.3, 0.8])
    lmd = 0.05
    T = 10000
    set_param(lmd=lmd, tau=tau, T_value=T, mu_value=mu, c_star_value=c_star)
    t = incentive()
    lst = explore(t=t)
    print(lst[0])
    exploit(arm=lst[0], t=lst[1])
    draw_plot()
    exit(0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
