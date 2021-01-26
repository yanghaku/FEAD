import numpy as np
from CNN_mawilab import CNN_mawilab


def get_bin_num(xx):
    ans = 0
    x = xx.copy()[0]
    for i in range(13):
        if x & 1 == 1:
            ans += 1
        x >>= 1
    return ans


test = CNN_mawilab()

f1 = 0
num = 14
now = 0
t = 0
T = 20
while t < T:
    feature = np.random.randint(1, 2 ** 13, 1)
    print(feature, end=' ')
    d = get_bin_num(feature)
    f = test.run(feature)
    print(feature, f)
    if f[0] > f1 or (f[0] == f1 and num > d):
        t = 0
        now = feature
        num = d
        f1 = f[0]
    else:
        t += 1
