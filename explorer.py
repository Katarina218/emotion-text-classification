import pandas as pd
df = pd.read_csv("archive\emotions.txt", sep = ";", header = None, names = ["text", "emotion"])
df["word_count"] = df["text"].apply(lambda x: len(x.split()))

import matplotlib.pyplot as plt

df["word_count"].hist(bins = 20)
plt.title("Rozlozenie dlzky textov")
plt.xlabel("pocet slov")
plt.ylabel("Pocet textov")
plt.tight_layout
plt.savefig("Rozlozenie_dlzky_textov.png")
plt.show()

print(df.groupby("emotion")["word_count"].mean())