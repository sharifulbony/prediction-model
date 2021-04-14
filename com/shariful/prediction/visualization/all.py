import seaborn as sns
import matplotlib.pyplot as plt

def all_visual(data):
    figure = plt.figure(figsize=(20, 8))
    sns.boxplot(x="age", y="outcome_pregnancy", data=data)
    plt.show()

    figure = plt.figure(figsize=(20, 8))
    sns.boxplot(x="alcohol", y="outcome_pregnancy", data=data)
    plt.show()

    figure = plt.figure(figsize=(20, 8))
    sns.barplot(x="smoke", y="outcome_pregnancy", data=data)
    plt.show()