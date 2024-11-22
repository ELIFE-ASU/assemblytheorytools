import copy
import json
import re

from assembly_pathv3 import assemblyConstruction
from log_parsing import v_string_convert, e_string_convert


def parse_str_log(file_name):
    """parse_str_log takes a file from the AssemblyGo String log output and outputs all relevant information for its pathway reconstruction

    :param file_name: input file location e.g. "/your_file_location"
    :return string: outputs the original string e.g. "aaaaaaaabbbbaaaaaaa"
    :return rem_pre_str: outputs the remant string list: ['aa','bb','a']
    :return rem_pre_id: outputs the remant string correspoinding original list location [0,16,18]
    :return repeated: outputs the duplicated strings [[[0, 6], [12, 18]],[[6, 8], [12, 14]],[[8, 10], [10, 12]],[[12, 14], [14, 16]],[[14, 16], [16, 18]]]
    :return runtime: outputs runtime
    """
    f = open(file_name, "r")
    counter = 0
    repeated = []
    for x in f:
        if x[-16:-1] == "ORIGINAL String":
            counter = -1
            continue
        if counter == -1:
            counter = -2
            continue
        if counter == -2:
            string = x[18:-2]
            counter = -3
            continue
        if x[0:14] == "Remnant String":
            counter = 1
            continue
        if counter == 1:
            rem_pre_str = v_string_convert(x[17:-1], "str")
            counter = 2
            continue
        if counter == 2:
            rem_pre_id = v_string_convert(x[19:-1], "int")
            counter = 3
            continue
        if x[0:18] == "Duplicated Strings":
            counter = 6
            continue
        if counter == 6 and x == "\n":
            counter = 7
            continue
        if counter == 6:
            repeated.append(e_string_convert(x[0:-1]))
        s = re.sub(r'[0-9]', " ", x)
        s = re.sub(r'\W+', " ", s).strip()
        if s == "Time":
            runtime = float(x[x.find("Time") + 7:])

    return string, rem_pre_str, rem_pre_id, repeated, runtime


def transform_string_mol(string, rem_pre_str, rem_pre_id, dup_pre):
    """transform_string_mol takes the pathway reconstruction information of an String and convert it to the path reconstruction information for a molecule that is a linear chain

    The string "aaaaaaaabbbbaaaaaaa" will be converted to the following molecule chain:
            
        p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p──►p
          a   a   a   a   a   a   a   a   b   b   b   b   a   a   a   a   a   a   a

    :param string: outputs the original string e.g. "aaaaaaaabbbbaaaaaaa"
    :param rem_pre_str: outputs the remant string list: ['aa','bb','a']
    :param rem_pre_id: outputs the remant string correspoinding original list location [0,16,18]
    :param repeated: outputs the duplicated strings [[[0, 6], [12, 18]],[[6, 8], [12, 14]],[[8, 10], [10, 12]],[[12, 14], [14, 16]],[[14, 16], [16, 18]]]
    :return v: vertices list [0, 1, 2, 3, 4, ... , 17, 18, 19]
    :return e: edges list [[0, 1],[1, 2],[2, 3],[3, 4],...,[17, 18],[18, 19]]
    :return v_l: vertex labels ['p','p','p','p',...,,'p','p','p']
    :return e_l: edges labels['a','a','a','a','a','a','a','a','b','b','b','b','a','a','a','a','a','a','a']
    :return remnant_e: remanant edges [[10, 11], [11, 12], [16, 17], [17, 18], [18, 19]]
    :return equivalences: compatibility of equivalences to molecule construction [[1,1]]
    :return duplicates: duplicates edges [[[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],[[12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18]]],[[[6, 7], [7, 8]], [[12, 13], [13, 14]]],[[[14, 15], [15, 16]], [[16, 17], [17, 18]]]]
    """
    v = list(range(len(string) + 1))
    e = [[i, i + 1] for i in v[:-1]]
    v_l = ["p" for _ in v]
    e_l = [string[i] for i, _ in enumerate(e)]
    remnant_e_pre = [[[item + j, item + j + 1] for j, _ in enumerate(rem_pre_str[i])] for i, item in
                     enumerate(rem_pre_id)]
    remnant_e = []
    for rem in remnant_e_pre:
        remnant_e = remnant_e + rem
    duplicates = [[[[index, index + 1] for index in range(pair[0], pair[1])] for pair in duplicate] for duplicate in
                  dup_pre]
    equivalences = [[1, 1]]
    return v, e, v_l, e_l, remnant_e, equivalences, duplicates


def generate_string_pathway(filename):
    """generate_string_pathway takes a file from the AssemblyGo String log output and outputs an assemblypath construction object 

    :param file_name: input file location e.g. "/your_file_location"
    :return construction_object: construction object that contains the pathway information of the string assembly
    """
    string, rem_pre_str, rem_pre_id, dup_pre, runtime = parse_str_log(filename)
    v, e, v_l, e_l, remnant_e, equivalences, duplicates = transform_string_mol(string, rem_pre_str, rem_pre_id, dup_pre)
    construction_object = assemblyConstruction(v, e, v_l, e_l, remnant_e, equivalences, duplicates, ifstring=True)
    try:
        construction_object.generate_pathway()
        steps_str = []
        for step in construction_object.steps:
            step_str = ""
            for edge in step:
                step_str = step_str + e_l[e.index(edge)]
            steps_str.append(step_str)
        construction_object.steps_str = steps_str
        pathway_success = True
    except ValueError as e:
        pathway_success = False
        print("error")
    except IndexError as e:
        pathway_success = False
        print("error")

    return construction_object, pathway_success


def generate_string_pathway_ian(file):
    f = open(file)
    data = json.load(f)
    string = data["file_graph"][0]['Fragments'][0]
    rem_pre_str = data["remnant"][0]['Fragments'][0].split(', ')
    rem_pre_id = data["remnant"][0]['Positions']
    dup_pre = [[[dup["Right"][0], dup["Right"][0] + dup["Right"][1]], [dup["Left"][0], dup["Left"][0] + dup["Left"][1]]]
               for dup in data["duplicates"]]
    v, e, v_l, e_l, remnant_e, equivalences, duplicates = transform_string_mol(string, rem_pre_str, rem_pre_id, dup_pre)
    construction_object = assemblyConstruction(v, e, v_l, e_l, remnant_e, equivalences, duplicates, ifstring=True)

    try:
        construction_object.generate_pathway()
        steps_str = []
        for step in construction_object.steps:
            step_str = ""
            for edge in step:
                step_str = step_str + e_l[e.index(edge)]
            steps_str.append(step_str)
        construction_object.steps_str = steps_str
        pathway_success = True
    except ValueError as e:
        pathway_success = False
        print("error")
    except IndexError as e:
        pathway_success = False
        print("error")

    return construction_object, pathway_success


def get_graph_string_explicit(construction_object):
    """get_graph_string takes an assemblypath construction object for strings and outputs the vertices and edges of the DAG(Directed Acylcic Graph) from the pathway 

    :param construction_object: construction object that contains the pathway information of the string assembly
    :return vertices,construction_object.digraph: Vertex and Edges explicit string information of the DAG(Directed Acylcic Graph) from the pathway 
    """
    vertex_dict = {}
    for i, a in enumerate(construction_object.atoms_list):
        if a[1] == "C":
            continue
        ato = "{}".format(a[1])
        vertex_dict["atom{}".format(i)] = ato

    # print("*Steps List*")
    for i, step_mol in enumerate(construction_object.steps_str):
        ato = ''.join(step_mol)
        vertex_dict["step{}".format(i + 1)] = ato
    vertices = []
    for i, _ in enumerate(construction_object.atoms_list):
        vertices.append(vertex_dict["atom{}".format(i)])
    for i, _ in enumerate(construction_object.steps):
        vertices.append(vertex_dict["step{}".format(i + 1)])
    digraph = []
    for edges in construction_object.digraph:
        digraph.append([vertex_dict[edges[0]], vertex_dict[edges[1]]])
    return vertices, digraph


def get_graph_string(construction_object):
    """get_graph_string takes an assemblypath construction object for strings and outputs the vertices and edges of the DAG(Directed Acylcic Graph) from the pathway 

    :param construction_object: construction object that contains the pathway information of the string assembly
    :return vertices,construction_object.digraph: Vertex and Edges infromation of the DAG(Directed Acylcic Graph) from the pathway 
    """
    vertices = []
    for i, _ in enumerate(construction_object.atoms_list):
        vertices.append("atom{}".format(i))
    for i, _ in enumerate(construction_object.steps):
        vertices.append("step{}".format(i + 1))
    return vertices, construction_object.digraph


def export_mathematica(construction_object, aspect_ratio=1):
    """export_mathematica takes an assemblypath construction object for strings and outputs a mathematica notebook containing the quiver representation of the assembly space 
       in the same directory as the python file. For Quiver information see: https://www.mdpi.com/1099-4300/24/7/884

    :param construction_object: construction object that contains the Assembly Space information of the string assembly 
    :return None:
    """
    pathway_file = open("pathway_string_mathematica.nb", "w")
    # print("*Atoms List*")
    pathway_file.write("stripMetadata[expression_] := If[Head[expression] === Rule, Last[expression], expression];\n")
    vertex_string = "v={"
    # print("*Atoms List*")
    str_atoms = ""
    for i, a in enumerate(construction_object.atoms_list):
        str_atoms = str_atoms + "atom{},".format(i)
        if a[1] == "C":
            continue
        ato = "{}".format(a[1])
        out = [i, ato]
        pathway_file.write("atom{} =  \"{}\" ;\n".format(*out))
        vertex_string = vertex_string + "atom{},".format(i)

    # print("*Steps List*")
    for i, step_mol in enumerate(construction_object.steps_str):
        ato = ''.join(step_mol)
        out = [i + 1, ato]
        pathway_file.write("step{} =   \"{}\" ;\n".format(*out))
        vertex_string = vertex_string + "step{},".format(i + 1)
    vertex_string = vertex_string[:-1] + "};"
    # print("*Vertex List*")
    pathway_file.write(vertex_string)
    dig = ""
    # print("*Digraph List*")
    digraph_temp = copy.deepcopy(construction_object.digraph)
    for i, el in enumerate(construction_object.digraph):
        if i % 2 == 0:
            digraph_temp[i].append(construction_object.digraph[i + 1][0])
        if i % 2 == 1:
            digraph_temp[i].append(construction_object.digraph[i - 1][0])
    for ed in digraph_temp:
        dig = dig + "Labeled[{} \[DirectedEdge] {},{}],".format(ed[0], ed[1], ed[2])
    pathway_file.write("e = {{ {} }};\n".format(dig[0:-1]))

    pathway_file.write(
        "elabeled = Table[Labeled[e[[i]][[1]], Placed[Text[ Framed[Style[stripMetadata[e[[i]][[2]]], Hue[0.62, 1, 0.48], FontSize -> 8], Background -> Directive[Opacity[0.2], Hue[0.1, 0.45, 0.87]], FrameMargins -> {{2, 2}, {0, 0}}, RoundingRadius -> 0, FrameStyle -> Directive[Opacity[0.5], Hue[0.62, 0.52, 0.82]]]], 1/2]], {i, Length[e]}];\n")

    pathway_file.write(
        "gim = Graph[v, elabeled, {AspectRatio -> 1, EdgeStyle -> {Directive[{Hue[0.62, 0.85, 0.87], Dashing[None], AbsoluteThickness[1]}]}, FormatType -> TraditionalForm, GraphLayout -> {\"Dimension\" -> 2}, PerformanceGoal -> \"Quality\", VertexShapeFunction -> {Text[ Framed[Style[stripMetadata[#2], Hue[0.62, 1, 0.48]], Background -> Directive[Opacity[0.2], Hue[0.62, 0.45, 0.87]], FrameMargins -> {{2, 2}, {0, 0}}, RoundingRadius -> 0, FrameStyle -> Directive[Opacity[0.5], Hue[0.62, 0.52, 0.82]]], #1, {0, 0}] &}}];\n")

    pathway_file.write("atoms = {" + str_atoms[:-1] + "};\n")
    pathway_file.write("LayeredGraphPlot[gim, atoms -> Automatic, AspectRatio -> {}] ".format(aspect_ratio))
    pathway_file.close()
    return None
