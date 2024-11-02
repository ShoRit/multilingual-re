import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# Sample dep_graph data for the example
# dep_graph = [
#     ["From", "case", "period"],
#     ["about", "advmod", "period"],
#     ["the", "det", "period"],
#     ["same", "amod", "period"],
#     ["period", "obl", "came"],
#     ["came", "root", "ROOT"],
#     ["the", "det", "poetry"],
#     ["romantic", "amod", "poetry"],
#     ["poetry", "obj", "came"],
#     ["of", "case", "Juan"],
#     ["Juan", "nmod", "poetry"],
#     ["Zorrilla", "flat", "Juan"],
#     ["de", "flat", "Juan"],
#     ["San", "flat", "Juan"],
#     ["Martín", "flat", "Juan"],
#     ["(", "punct", "1855"],
#     ["1855", "nmod:tmod", "Juan"],
#     ["–1931", "nummod", "1855"],
#     [")", "punct", "1855"],
#     [",", "punct", "wrote"],
#     ["who", "nsubj", "wrote"],
#     ["wrote", "acl:relcl", "Juan"],
#     ["epic", "amod", "poems"],
#     ["poems", "obj", "wrote"],
#     ["about", "case", "history"],
#     ["Uruguayan", "amod", "history"],
#     ["history", "nmod", "poems"],
#     [".", "punct", "came"]
# ]

# # Define e1 and e2 entities as lists of possible target words
# e1 = ["Zorrilla", "de", "Juan", "San", "Martín"]
# e2 = ["romantic","poetry"]

dep_graph= [
                [
                    "A",
                    "det",
                    "drop"
                ],
                [
                    "lemon",
                    "compound",
                    "drop"
                ],
                [
                    "drop",
                    "nsubj",
                    "candy"
                ],
                [
                    "is",
                    "cop",
                    "candy"
                ],
                [
                    "a",
                    "det",
                    "candy"
                ],
                [
                    "sugar",
                    "obl:npmod",
                    "coated"
                ],
                [
                    "coated",
                    "amod",
                    "candy"
                ],
                [
                    ",",
                    "punct",
                    "candy"
                ],
                [
                    "lemon",
                    "compound",
                    "flavored"
                ],
                [
                    "-",
                    "punct",
                    "flavored"
                ],
                [
                    "flavored",
                    "amod",
                    "candy"
                ],
                [
                    "candy",
                    "root",
                    "ROOT"
                ],
                [
                    "that",
                    "nsubj:pass",
                    "colored"
                ],
                [
                    "is",
                    "aux:pass",
                    "colored"
                ],
                [
                    "typically",
                    "advmod",
                    "colored"
                ],
                [
                    "colored",
                    "acl:relcl",
                    "candy"
                ],
                [
                    "yellow",
                    "xcomp",
                    "colored"
                ],
                [
                    "and",
                    "cc",
                    "shaped"
                ],
                [
                    "often",
                    "advmod",
                    "shaped"
                ],
                [
                    "shaped",
                    "conj",
                    "colored"
                ],
                [
                    "like",
                    "case",
                    "lemon"
                ],
                [
                    "a",
                    "det",
                    "lemon"
                ],
                [
                    "miniature",
                    "amod",
                    "lemon"
                ],
                [
                    "lemon",
                    "obl",
                    "shaped"
                ],
                [
                    ".",
                    "punct",
                    "candy"
                ]
]

e1=["lemon", "drop"]
e2=["yellow"]

# Create directed graph
G = nx.DiGraph()

# Add nodes and edges with labels
for relation in dep_graph:
    target, label, source = relation
    G.add_edge(target, source, label=label)

print(G)


# BFS function to find the first entry point for an entity starting from the root
def find_first_entry_point(graph, root, entity_words):
    queue = deque([root])
    visited = set()
    
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        
        # Check if the node is part of the entity words
        if node in entity_words:
            return node
        
        # Add neighbors to the queue for BFS
        queue.extend(graph.predecessors(node))
    
    return None  # If no entry point is found

# Find first entry points for e1 and e2 from the root
root = "ROOT"
first_entry_e1 = find_first_entry_point(G, root, e1)
first_entry_e2 = find_first_entry_point(G, root, e2)

print(first_entry_e1)
print(first_entry_e2)

print(G)

try:
    # Try finding a direct path between first_entry_e1 and first_entry_e2
    path_e1_to_e2 = nx.shortest_path(G, source=first_entry_e1, target=first_entry_e2)
    highlighted_path = path_e1_to_e2  # Direct path found
except nx.NetworkXNoPath:
    try:
        path_e1_to_e2 = nx.shortest_path(G, source=first_entry_e2, target=first_entry_e1)
        highlighted_path = path_e1_to_e2  # Direct path found
    except nx.NetworkXNoPath:
        try:
            path_root_to_e1 = nx.shortest_path(G, source=first_entry_e1, target=root)
            try:
                path_e1_to_root = nx.shortest_path(G, source=root, target=first_entry_e1)
                if len(path_root_to_e1<path_e1_to_root):
                    highlighted_path=path_root_to_e1
                else:
                    highlighted_path=path_e1_to_root
            except:
                highlighted_path=path_root_to_e1
        except:
            pass

        try:
            path_root_to_e2 = nx.shortest_path(G, source=first_entry_e2, target=root)
            try:
                path_e2_to_root = nx.shortest_path(G, source=root, target=first_entry_e2)
                if len(path_root_to_e2<path_e2_to_root):
                    highlighted_path+=path_root_to_e2
                else:
                    highlighted_path+=path_e2_to_root
            except:
                highlighted_path+=path_root_to_e2
        except:
            pass



print("Highlighted path:", highlighted_path)


# Set up plot
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, seed=25)  # Positions for all nodes

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="lightblue")
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray")

# Draw the highlighted path in a different color, only if the edge exists in the original graph
if highlighted_path:
    edges_in_path = [
        (highlighted_path[i], highlighted_path[i + 1]) 
        for i in range(len(highlighted_path) - 1) 
        if G.has_edge(highlighted_path[i], highlighted_path[i + 1]) or G.has_edge(highlighted_path[i + 1], highlighted_path[i])
    ]
    nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color="red", width=2, arrowstyle="->", arrowsize=25)


# Draw labels for nodes and edges
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# Display the plot
plt.title("Dependency Graph with Highlighted Path from e1 to e2 Entry Points")
plt.axis("off")  # Hide axes
plt.show()
