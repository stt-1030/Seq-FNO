import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
def potential_energy(x, y, lambda_=1.0):
    """计算势能 V(x, y)"""
    return 0.5 * (x ** 2 + y ** 2) + lambda_ * (x ** 2 * y - y ** 3 / 3)

def generate_initial_conditions(H, num_samples, lambda_=1.0):
    """生成初值 [x, y, p_x, p_y]，满足总能量 H"""
    initial_conditions = []
    for _ in range(num_samples):
        while True:
            # 随机生成 x 和 y
            x = 0
            y = np.random.uniform(-1, 1)
            # 计算势能
            V = potential_energy(x, y, lambda_)
            # 如果势能超过 H，则重新生成
            if V >= H:
                continue
            # 随机生成 p_y
            py = np.random.uniform(-1, 1)
            # 计算 p_x
            px_squared = 2 * (H - V) - py ** 2
            if px_squared >= 0:
                px = np.sqrt(px_squared)
                initial_conditions.append([x, y, px, py])
                break  # 找到一个有效初值，退出循环
    return np.array(initial_conditions)

H = 1/15
num_samples = 3000
initial_conditions = generate_initial_conditions(H, num_samples)
print(initial_conditions.shape)
io.savemat('50_step_pred/IC.mat', {'ic':initial_conditions})
