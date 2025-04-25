def reduce(sig, n):
    new_sig = [min(sig[0], n)]
    for i in range(1, len(sig)):
        new_sig.append(min(sig[i], n - sum(new_sig)))
    return new_sig


def extend(sig, q, D):
    new_sig = [sig[0] + sig[1]] + sig[2:] + [0]
    x = ([-1] + D)
    for i in range(len(new_sig)):
        new_sig[i] = new_sig[i] + q * x[i]
    return new_sig

if __name__ == '__main__':
    n = 10
    c = [1,1,2]
    D = [2, 1]
    root = [0,2,1]
    s_1 = extend(root, 2, D)
    print(f"s_1: {s_1}")
    s_2 = extend(s_1, 3, D)
    print(f"s_2: {s_2}")
    s_2 = reduce(s_2, n)
    print(f"s_2: {s_2}")
    s_2_ = extend(s_1, 5, D)
    print(f"s_2_: {s_2_}")
    s_2_ = reduce(s_2_, n)
    print(f"s_2_: {s_2_}")



    s_3 = extend(s_2, 1, D)
    print(f"s_3: {s_3}", reduce(s_3, n))
    s_3_ = extend(s_2, 7, D)
    print(f"s_3_: {s_3_}", reduce(s_3_, n))
