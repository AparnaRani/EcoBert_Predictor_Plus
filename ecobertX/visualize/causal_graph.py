import networkx as nx
import matplotlib.pyplot as plt


def plot_causal_graph(trace, save_path):

    G = nx.DiGraph()

    for t in trace:

        G.add_edge(
            t["from"],
            t["to"],
            weight=t["strength"]
        )

    plt.figure(figsize=(10,6))

    pos = nx.spring_layout(G)

    nx.draw(

        G,
        pos,

        with_labels=True,

        node_color="lightblue",

        edge_color="gray",

        node_size=2000,

        font_size=9

    )

    plt.title("EcoBERT-X Mechanistic Causal Graph")

    plt.savefig(save_path)

    plt.close()

    print("Causal graph saved:", save_path)
