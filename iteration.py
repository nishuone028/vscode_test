def for_loop(n: int) -> int:
    res = 0 
    for i in range(1, n+1):
        res += i
    return res


def while_loop(n: int) -> int:
    res = 0
    i = 1
    while i <= n:
        res += i
        i += 1
    return res

def while_loop_ii(n: int) -> int:
    res = 0
    i = 1
    while i <= n:
        res += i
        i += 1
        i *= 2
    return res

def nested_loops(n: int) -> int:
    res = ""
    for i in range(1, n+1):
        for j in range(1, n+1):
            res += f"{i}, {j}, "
    return res

def recur(n: int) -> int:
    """递归"""
    # 终止条件
    if n == 1:
        return 1
    # 递：递归调用
    res = recur(n - 1)
    # 归：返回结果
    return n + res

if __name__ == '__main__':
    print(recur(5))
    # print(nested_loops(10))
    # print(while_loop_ii(10))
    # print(while_loop(10))
    # print(for_loop(10))