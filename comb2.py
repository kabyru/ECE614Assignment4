def comb(x):
    import itertools
    combinations = []
    combinations.extend(itertools.combinations_with_replacement(x,3))

    combinationsCopy = combinations.copy()

    print(combinations)

    for i in range(0,len(combinations)):
        reverseTuple = tuple(reversed(combinations[i]))
        #print(reverseTuple)
        if (reverseTuple == combinations[i]):
            continue
        else:
            combinationsCopy.append(reverseTuple)

    combinations = combinationsCopy

    for k in range(0,len(combinations)):
        combinations[k] = list(combinations[k])

    return combinations

