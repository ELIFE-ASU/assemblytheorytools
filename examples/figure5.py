import networkx as nx
import assemblytheorytools as att
import matplotlib.pyplot as plt

# This script will make the circuit diagram figure for the ATT paper.

def make_n_bit_adder(full_adder, n):
    """
    Constructs an N-bit adder by stitching together N copies of a full adder graph.
    The C_out of adder i is contracted with the C_in of adder i+1.
    The A, B, S tags are relabeled to A_i, B_i, S_i for each bit index i.
    """
    adders = []
    tag_map = {}
    for i in range(n):
        # Create a copy of the full adder
        G = nx.relabel_nodes(full_adder, lambda x: x + i*100, copy=True)
        # Relabel tags
        for node, data in G.nodes(data=True):
            if 'tag' in data:
                if data['tag'] == 'A':
                    data['tag'] = f'A_{i}'
                elif data['tag'] == 'B':
                    data['tag'] = f'B_{i}'
                elif data['tag'] == 'S':
                    data['tag'] = f'S_{i}'
        adders.append(G)
        # Record C_in and C_out node ids for stitching
        for node, data in G.nodes(data=True):
            if 'tag' in data:
                if data['tag'] == 'C_in':
                    tag_map[f'C_in_{i}'] = node
                elif data['tag'] == 'C_out':
                    tag_map[f'C_out_{i}'] = node

    # Start with the first adder
    result = adders[0]
    for i in range(1, n):
        # Merge the next adder into the result
        result = nx.compose(result, adders[i])
        # Contract C_out of previous with C_in of current
        contracted_node = tag_map[f'C_out_{i-1}']
        other_node = tag_map[f'C_in_{i}']
        result = nx.contracted_nodes(
            result,
            contracted_node,
            other_node,
            self_loops=False
        )
        # After contraction, retag the contracted carry node as C_i
        if 'tag' in result.nodes[contracted_node]:
            result.nodes[contracted_node]['tag'] = f'C_{i}'
        else:
            result.nodes[contracted_node]['tag'] = f'C_{i}'
    return result


# New version of the circuit assembly space:
full_adder = nx.Graph()
full_adder.add_node(0, color='w', tag='A')
full_adder.add_node(1, color='w', tag='B')
full_adder.add_node(2, color='X')
full_adder.add_edge(0, 2, color=1)
full_adder.add_edge(1, 2, color=1)
full_adder.add_node(3, color='A')
full_adder.add_edge(0, 3, color=1)
full_adder.add_edge(1, 3, color=1)
full_adder.add_node(4, color='w')
full_adder.add_edge(2, 4, color=2)
full_adder.add_node(5, color='w')
full_adder.add_edge(3, 5, color=2)
full_adder.add_node(6, color='X')
full_adder.add_edge(4, 6, color=1)
full_adder.add_node(7, color='w', tag='C_in')
full_adder.add_edge(7, 6, color=1)
full_adder.add_node(8, color='A')
full_adder.add_edge(4, 8, color=1)
full_adder.add_edge(7, 8, color=1)
full_adder.add_node(9, color='O')
full_adder.add_edge(5, 9, color=1)
full_adder.add_node(10, color='w', tag='S')
full_adder.add_edge(6, 10, color=2)
full_adder.add_node(11, color='w')
full_adder.add_edge(8, 11, color=2)
full_adder.add_edge(9, 11, color=1)
full_adder.add_node(12, color='w', tag='C_out')
full_adder.add_edge(9, 12, color=2)

plt.figure()
pos = nx.spring_layout(full_adder, seed=42)
labels = {n: full_adder.nodes[n].get('color', n) for n in full_adder.nodes}
nx.draw(full_adder, pos, with_labels=True, labels=labels, node_color='lightblue', edge_color='gray', node_size=500)
edge_labels = {(u, v): full_adder.edges[u, v].get('color', '') for u, v in full_adder.edges}
nx.draw_networkx_edge_labels(full_adder, pos, edge_labels=edge_labels)
plt.title(f"Full Adder Circuit")
plt.savefig(f"full_adder.png", dpi=300)
plt.close()

two_bit_adder_v2 = make_n_bit_adder(full_adder, 2)

for i, obj in enumerate([full_adder, two_bit_adder_v2]):
    obj = att.canonicalize_node_labels(obj)
    ai, vo, path, logfile = att.calculate_assembly_index(obj, return_log_file=True, debug=True)
    print(f"Assembly Index for {i+1}bit adder = {ai}")
    print(f"Object size = {len(obj.edges())}\n")
    if i == 0:
        for idx, g in enumerate(vo):
            plt.figure()
            pos = nx.spring_layout(g, seed=42)
            labels = {n: g.nodes[n].get('color', n) for n in g.nodes}
            nx.draw(g, pos, with_labels=True, labels=labels, node_color='lightblue', edge_color='gray', node_size=500)
            edge_labels = {(u, v): g.edges[u, v].get('color', '') for u, v in g.edges}
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
            plt.title(f"vo[{idx}]")
            plt.savefig(f"full_adder_circuit_vo_{idx}.png", dpi=300)
            plt.close()
    


