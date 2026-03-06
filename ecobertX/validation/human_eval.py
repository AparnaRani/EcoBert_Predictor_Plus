import pandas as pd

def analyze_human_eval(file):

    df = pd.read_csv(file)

    results = {
        "clarity": df["Clarity"].mean(),
        "usefulness": df["Usefulness"].mean(),
        "interpretability": df["Interpretability"].mean()
    }

    return results