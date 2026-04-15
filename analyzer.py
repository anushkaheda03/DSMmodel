import pandas as pd
def process_data(df, cutoff):
    df["Pass"] = df["marks"] >= cutoff

    def grade(m):
        if m >= 90:
            return "Distinction"
        elif m >= 75:
            return "First Class"
        elif m >= cutoff:
            return "Pass"
        else:
            return "Fail"

    df["Grade"] = df["marks"].apply(grade)
    return df