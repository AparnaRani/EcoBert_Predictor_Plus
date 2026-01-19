def interpret_prediction_dual(raw_row, explainer):

    pred = explainer.predict(raw_row)
    reasons = explainer.explain_prediction(raw_row)

    print("\n===================================")
    print(f"🔹 Predicted CO₂: {pred:.5f} kg")
    print("===================================\n")

    # Remove unwanted columns
    reasons = [r for r in reasons if "Unnamed" not in r["feature"]]

    # Sort by importance
    reasons = sorted(reasons, key=lambda x: abs(x["impact"]), reverse=True)

    # ---------- SCIENTIFIC ----------
    print("📘 SCIENTIFIC INTERPRETATION\n")

    for r in reasons[:10]:
        sign = "↑ increases" if r["impact"] > 0 else "↓ decreases"
        print(f"{r['feature']:<25} {sign} prediction by {abs(r['impact']):.4f}")

    # ---------- FRIENDLY ----------
    print("\n😊 HUMAN-FRIENDLY EXPLANATION\n")

    for r in reasons[:10]:
        verb = "raised emissions by" if r["impact"] > 0 else "lowered emissions by"
        print(f"- {r['feature']} {verb} {abs(r['impact']):.4f}")

    # ---------- SUMMARY ----------
    inc = [r for r in reasons if r["impact"] > 0]
    dec = [r for r in reasons if r["impact"] < 0]

    print("\n🧠 SUMMARY\n")

    if dec:
        print("Main factors REDUCING emissions:")
        for r in dec[:3]:
            print(" •", r["feature"])

    if inc:
        print("\nMain factors INCREASING emissions:")
        for r in inc[:3]:
            print(" •", r["feature"])

    print("\n📗 Responsible AI Note:")
    print(
        "This explanation converts the model's internal reasoning "
        "into both scientific and human-readable forms, supporting "
        "transparent carbon-aware NLP training."
    )
