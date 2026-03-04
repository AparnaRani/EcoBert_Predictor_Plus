import matplotlib.pyplot as plt

def plot_interpretation(reasons, predicted_co2, save_path):
    """
    reasons: output from xai.explain_prediction()
    """

    features = [r["feature"] for r in reasons]
    impacts  = [r["impact"] for r in reasons]

    colors = ["green" if i < 0 else "red" for i in impacts]

    plt.figure(figsize=(9,6))
    plt.barh(features, impacts, color=colors)
    plt.axvline(0, color="black", linewidth=1)

    plt.title(f"EcoBERT-X Explanation\nPredicted CO₂ = {predicted_co2:.5f} kg")
    plt.xlabel("Impact on CO₂ (kg)")
    plt.ylabel("Features")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("📊 Visualization saved →", save_path)
