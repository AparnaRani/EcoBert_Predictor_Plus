import matplotlib.pyplot as plt


def plot_heatmap(explanation, save_path):

    explanation = [
        e for e in explanation
        if "Unnamed" not in str(e["feature"])
    ]

    features = [e["feature"] for e in explanation[:15]]
    impacts = [e["impact"] for e in explanation[:15]]

    plt.figure(figsize=(8,6))

    plt.barh(features, impacts)

    plt.title("EcoBERT-X Feature Impact Heatmap")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()
