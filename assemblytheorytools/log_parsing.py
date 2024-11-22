import re

import numpy as np


def v_string_convert(v_string, input_type):
    """v_string_convert transforms a string "[0 1 2]" into a list of ints of strings [0, 1, 2]

    :param v_string: input string "[0 1 2]" or "[A B C]"
    :param input_type: use either a integer or string conversion
    :return: outputs a list [0, 1, 2] or ["A", "B", "C"]
    """
    indices = []
    if input_type == "int":
        f = int
    if input_type == "str":
        f = str
    for i, c in enumerate(v_string):
        if c == " ":
            indices.append(i)
    if indices != []:
        v = [f(v_string[1:indices[0]])]
        for i, j in enumerate(indices):
            if i == len(indices) - 1:
                break
            v.append(f(v_string[indices[i] + 1:indices[i + 1]]))
    else:
        indices = [0]
        v = []
    v.append(f(v_string[indices[-1] + 1:-1]))
    return v


def e_string_convert(e_string):
    """e_string_convert transforms a string "[[0 1] [1 2]]" into a list of list of ints [[0, 1],[1,2]]

    :param e_string: input string "[[0 1] [1 2]]"
    :return: outputs a list of list of ints [[0, 1],[1,2]]
    """
    export = np.fromstring(e_string.replace("[", "").replace("]", ""), dtype=int, sep=' ')
    e = export.reshape((int(len(export) / 2), 2)).tolist()
    return e


def dup_string_convert(e_string):
    """dup_string_convert transforms a string "[[1 2] [3 4]] [[5 6] [7 8]]" into a list of list of ints [[[1, 2],[3, 4]],[[5, 6],[7, 8]]]

    :param e_string: input string "[[1 2] [3 4]] [[5 6] [7 8]]"
    :return: outputs an ordered pair of list of list of ints [[[1, 2],[3, 4]],[[5, 6],[7, 8]]]
    """
    export = np.fromstring(e_string.replace("[", "").replace("]", ""), dtype=int, sep=' ')
    final = export.reshape((2, int(len(export) / 2)))
    dup = [final[0].reshape((int(len(final[0]) / 2), 2)).tolist(),
           final[1].reshape((int(len(final[1]) / 2), 2)).tolist()]
    return dup


def parse_log(file_name):
    """parse_log takes a file from the AssemblyGo log output and outputs all relevant information for its pathway reconstruction

    :param e_string: input file location e.g. "/your_file_location"
    :return v: outputs a list of ints for the original graph vertex labels [0, 1, 2]
    :return e: outputs a list of list of ints for the original graph edge labels [[0, 1],[1,2]]
    :return v_c: outputs a list of strings for the original graph vertex Atoms ["A", "B", "C"]
    :return e_c: outputs a list ofof list strings for the original graph vertex bonds ["single", "double", "single"]
    :return vr: outputs a list of ints for the remnant graph vertex labels [0, 1, 2]
    :return er: outputs a list of list of ints for the remnant graph edge labels [[0, 1],[1,2]]
    :return vr_c: outputs a list of strings for the remnant graph vertex Atoms ["A", "B", "C"]
    :return er_c: outputs a list ofof list strings for the remnant graph vertex bonds ["single", "double", "single"]
    :return repeated: outputs the total repeated ordered pairs of list of list of ints [[[[1, 2],[3, 4]],[[5, 6],[7, 8]]],[[[9, 10],[11, 12]],[[13, 14],[15, 16]]]]
    :return equivalences: outputs the total repeated ordered pairs of repeated vertex labels[[0, 1],[1,2]]
    """
    f = open(file_name, "r")
    counter = 0
    repeated = []
    equivalences = []
    for x in f:
        if x[-15:-1] == "ORIGINAL GRAPH":
            counter = -1
            continue
        if counter == -1:
            counter = -2
            continue
        if counter == -2:
            v = v_string_convert(x[9:-1], "int")
            counter = -3
            continue
        if counter == -3:
            e = e_string_convert(x[6:-1])
            counter = -4
            continue
        if counter == -4:
            v_c = v_string_convert(x[14:-1], "str")
            counter = -5
            continue
        if counter == -5:
            e_c = v_string_convert(x[12:-2], "str")
            counter = -6
            continue
        if x[0:13] == "Remnant Graph":
            counter = 1
            continue
        if counter == 1:
            vr = v_string_convert(x[9:-1], "int")
            counter = 2
            continue
        if counter == 2:
            er = e_string_convert(x[6:-1])
            counter = 3
            continue
        if counter == 3:
            vr_c = v_string_convert(x[14:-1], "str")
            counter = 4
            continue
        if counter == 4:
            er_c = v_string_convert(x[12:-2], "str")
            counter = 5
            continue
        if x[0:16] == "Duplicated Edges":
            counter = 6
            continue
        if counter == 6 and x[0:15] == "+++++++++++++++":
            counter = 7
            continue
        if counter == 6:
            repeated.append(dup_string_convert(x[1:-2]))
        if x[0:16] == "Atom Equivalents":
            counter = 8
            continue
        if counter == 8 and x[0:15] == "###############":
            counter = 9
            continue
        if counter == 8:
            equivalences.append(v_string_convert(x[:-1], 'int'))
        s = re.sub(r'[0-9]', " ", x)
        s = re.sub(r'\W+', " ", s).strip()
        if s == "Time":
            runtime = float(x[x.find("Time") + 7:])

    return v, e, v_c, e_c, vr, er, vr_c, er_c, repeated, equivalences, runtime


def print_array(array):
    outp = ''
    for a in array:
        outp = outp + str(a).replace("'", "").replace("[", "").replace("]", "|")

    return outp


def print_array_2(array):
    outp = ''
    for a in array:
        for e in a:
            for b in e:
                outp = outp + str(b).replace("'", "").replace("[", "").replace("]", "|")
            outp = outp + '|'
        outp = outp + '|'
    return outp


def encode_path_data(file_name):
    v, e, v_c, e_c, vr, er, vr_c, er_c, repeated, equivalences, runtime = parse_log(file_name)
    # if not os.path.exists('paths_encoded'):
    #    os.makedirs('paths_encoded')
    with open('encoded_{}'.format(file_name), 'w') as f:
        f.write("10\n")
        f.write(str(runtime) + "\n")
        f.write(print_array(e).replace(" ", "")[:-1])
        f.write("\n")
        f.write(''.join(v_c))
        f.write("\n")
        atom_bonds = [bond for bond in set(e_c)]
        e_c_comp = ["{}".format(atom_bonds.index(bond)) for bond in e_c]
        for i, bond in enumerate(atom_bonds):
            f.write("{}{}".format(i, bond))
            f.write("\n")
        f.write(''.join(e_c_comp))
        f.write("\n")
        f.write(print_array(er).replace(" ", "")[:-1])
        f.write("\n")
        er_c_comp = ["{}".format(atom_bonds.index(bond)) for bond in er_c]
        f.write(''.join(er_c_comp))
        f.write("\n")
        f.write(print_array_2(repeated).replace(" ", "")[:-3])
        f.write("\n")
        f.write(print_array(equivalences).replace(" ", "")[:-1])

    return None
