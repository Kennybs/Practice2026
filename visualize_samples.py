import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/PreparedData.csv", sep=";", low_memory=False)
df = df.drop(columns=["Unnamed: 9"], errors="ignore")
df.columns = df.columns.str.strip()
import matplotlib.pyplot as plt

for sample_id, sample in df.groupby("sample_id"):

    sample = sample.reset_index(drop=True)

    peak_index = sample["Напряжение"].idxmax()

    peak_strain = sample.loc[peak_index, "Деформация"]
    final_strain = sample.iloc[-1]["Деформация"]

    if final_strain < peak_strain:

        plt.figure(figsize=(6,4))
        plt.plot(sample["Деформация"], sample["Напряжение"])

        plt.scatter(
            peak_strain,
            sample.loc[peak_index, "Напряжение"],
            color="red",
            label="Peak"
        )

        plt.scatter(
            final_strain,
            sample.iloc[-1]["Напряжение"],
            color="green",
            label="Final"
        )

        plt.title(sample_id)
        plt.legend()
        plt.show()