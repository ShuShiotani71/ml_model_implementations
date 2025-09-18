import copy

def partitions(elements, k):
    """
    compute all possible ways of partitioning a set into k subsets
    """
    if len(elements) == 0 and k == 0:
        return [[]]
    if len(elements) == 0 or k == 0:
        return []

    first, rest = elements[0], elements[1:]
    result = []

    # Case 1: put first element into existing subsets where
    # each one is of size k
    for partition in partitions(rest, k):
        for i in range(len(partition)):
            new_partition = copy.deepcopy(partition)
            new_partition[i].append(first)
            result.append(new_partition)

    # Case 2: put first element into its own new group where
    # each one is of size k-1
    for partition in partitions(rest, k - 1):
        new_partition = copy.deepcopy(partition)
        new_partition.append([first])
        result.append(new_partition)

    return result

def powerset(items):
    """
    compute the powerset
    """
    powersets = [[]]
    for item in items:
        tmp = []
        for powerset in powersets:
            tmp.append(powerset + [item])
        powersets += tmp
    return powersets
