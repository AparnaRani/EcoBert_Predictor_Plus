from explain.causal_reasoning import generate_causal_reason


def interpret_prediction_full(raw_row, explainer):

    pred = explainer.predict(raw_row)

    explanation = explainer.explain_prediction_detailed(raw_row)

    confidence = explainer.confidence_score(raw_row)

    print("\n===================================")
    print("EcoBERT-X MECHANISTIC EXPLANATION")
    print("===================================\n")

    print(f"Predicted CO₂: {pred:.6f} kg")
    print(f"Confidence Score: {confidence:.3f}\n")

    print("Causal Explanation:\n")

    for e in explanation[:10]:

        reason = generate_causal_reason(
            e["feature"],
            e["value"],
            e["impact"]
        )

        print("•", reason)
