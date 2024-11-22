import re

import numpy as np


def edgesfromstring(string):
    e = []
    couple = ""
    for i in string:
        couple = couple + i
        if i == "|":
            e.append(np.fromstring(couple, dtype=int, sep=","))
            couple = ""
    e.append(np.fromstring(couple, dtype=int, sep=","))

    return np.array(e, dtype=object)


def parse_db_path(file_name):
    f = open(file_name, "r")
    counter = 0
    line5 = 0
    bonds = []
    for i, x in enumerate(f):
        if i < 2:
            continue
        if i == 2:
            e = edgesfromstring(x)
            v = np.array(range(np.max(e) + 1))
            continue
        if i == 3:
            v_c = re.findall("[A-Z][^A-Z]*", x[:-1])
            continue
        if i > 3 and re.sub("\d+", "", re.sub("[^\w\s]", "", x)) != "\n" and line5 != 4:
            bonds.append(re.sub("\d+", "", x[:-1]))
            counter = counter + 1
            continue
        if counter > 0 and line5 == 0:
            e_c = [bonds[int(i)] for i in x[:-1]]
            line5 = line5 + 1
            continue
        if line5 == 1:
            er = edgesfromstring(x)
            line5 = line5 + 1
            continue
        if line5 == 2:
            line5 = line5 + 1
            continue
        if line5 == 3:
            split = [couple.split("||") for couple in x.split("|||")]
            repeated = []
            if split == [["\n"]]:
                equivalences = np.array([])
                continue
            for duple in split:
                repeated.append(
                    [
                        edgesfromstring(duple[0]).tolist(),
                        edgesfromstring(duple[1]).tolist(),
                    ]
                )
            line5 = line5 + 1
            equivalences = np.array([])
            continue
        if line5 == 4:
            equivalence = edgesfromstring(x)
            additional = []
            indices = []
            for i, equiv in enumerate(equivalence):
                if len(equiv) > 2:
                    first = equiv[0]
                    for ind in equiv[1:]:
                        additional.append([first, ind])
                    indices.append(i)
            additional = np.array(additional)
            new = np.delete(equivalence, indices, 0)
            if len(additional) != 0:
                #    new = np.stack(new, axis=0)
                #    add = additional
                #    for i,rep in enumerate(additional[:,1]):
                #        if rep in new[:,1]:
                #            add = np.delete(additional, i, 0)

                if new.size == 0:
                    equivalences = additional
                else:
                    equivalences = np.concatenate(
                        (np.array([i for i in new]), additional), axis=0
                    )
            else:
                equivalences = equivalence
            continue
    return (
        v.tolist(),
        e.tolist(),
        v_c,
        e_c,
        er.tolist(),
        equivalences.tolist(),
        repeated,
    )
