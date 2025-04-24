a =

def is_prefixfree(codes):
    """
    Check if the codes are prefix free
    :param codes:
    :return:
    """
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            if codes[i][0] == codes[j][0][:len(codes[i][0])]:
                print(codes[i])
                print(codes[j])
    return True

print(is_prefixfree(a))

# test if präfixfrei
for i in range(len(a)):
    for j in range(len(a)):
        if i == j:
            continue
        if a[i][0] == a[j][:len(a[i][0])]:
            print(a[i])
            print("Falsch")
            break