import networkx as nx
import assemblytheorytools as att


def test_assemblydepth():
    print(flush=True)
    G = nx.DiGraph()
    
    node_to_level = {
        "CC": 0,
        "C=C": 0,
        "CO": 0,
        "CC=C": 1,
        "OCC=C": 2
    }
    
    G.add_node("CC") # AD 0
    G.add_node("C=C") # AD 0
    G.add_node("CO") # AD 0
    G.add_node("CC=C") # AD 1
    G.add_node("OCC=C") # AD 2
    
    G.add_edge("CC", "CC=C")
    G.add_edge("C=C", "CC=C")
    
    G.add_edge("CO", "OCC=C")
    G.add_edge("CC=C", "OCC=C")

    att.assign_levels(G)

    for node, level in node_to_level.items():
        assert G.nodes[node]["level"] == level, f"Node {node} has incorrect level: {G.nodes[node]['level']} instead of {level}"
    print("All assembly depth tests passed.", flush=True)
    
def test_assemblydepth_linear_chain():
    print(flush=True)
    G = nx.DiGraph()
    
    node_to_level = {
        "CC": 0,  # AD 0
        "CCC": 1, # AD 1
        "CCCCC": 2, # AD 2
        "CCCCCCCCC": 3  # AD 3
    }
    
    G.add_node("CC")  # AD 0
    G.add_node("CCC")  # AD 1
    G.add_node("CCCCC")  # AD 2
    G.add_node("CCCCCCCCC")  # AD 3
    
    G.add_edge("CC", "CCC")
    G.add_edge("CCC", "CCCCC")
    G.add_edge("CCCCC", "CCCCCCCCC")

    att.assign_levels(G)

    for node, level in node_to_level.items():
        assert G.nodes[node]["level"] == level, f"Node {node} has incorrect level: {G.nodes[node]['level']} instead of {level}"
    print("Linear chain assembly depth test passed.", flush=True)

def test_assemblydepth_empty_graph():
    print(flush=True)
    G = nx.DiGraph()
    
    att.assign_levels(G)
    
    assert len(G.nodes) == 0, "Empty graph should have no nodes."
    print("Empty graph assembly depth test passed.", flush=True)
    

