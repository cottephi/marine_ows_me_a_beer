import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

columns = ["SEXE", "HEMOGLOBINE_PREOP", "DATE.INDUC", "DATE.DEATH", "INHOSPITAL.death", "ALIVE.J30"]


def get_data():
    data = pd.read_csv("POSE.csv", index_col=0, parse_dates=["DATE.INDUC", "DATE.DEATH"]).loc[:, columns]
    print(f"There are {len(data)} patients")
    data = data.replace("OUI", True)
    data = data.replace("NON", False)
    data = data.loc[~data["HEMOGLOBINE_PREOP"].isna().values]
    print(f"There are {len(data)} patients with available Hb data")
    return data


def set_alive(data):
    data["ALIVE"] = pd.Series(np.nan, index=data.index)

    alive_yes = data.loc[data["ALIVE.J30"].apply(lambda x: x is True)].index
    alive_no = data.loc[data["ALIVE.J30"].apply(lambda x: x is False)].index

    print(f"There are {len(alive_yes)} patients that are still alive from ALIVE.J30")
    print(f"There are {len(alive_no)} patients that are dead from ALIVE.J30")

    data.loc[alive_yes, "ALIVE"] = True
    data.loc[alive_no, "ALIVE"] = False
    index_dead_from_date = data[(data["DATE.DEATH"] - data["DATE.INDUC"]).apply(lambda x: x.days < 30)].index

    index_dead_from_date = data.loc[index_dead_from_date][
        data.loc[index_dead_from_date, "ALIVE.J30"].apply(lambda x: pd.isna(x))
    ].index
    index_alive_from_date = data[(data["DATE.DEATH"] - data["DATE.INDUC"]).apply(lambda x: x.days >= 30)].index
    index_alive_from_date = data.loc[index_alive_from_date][
        data.loc[index_alive_from_date, "ALIVE.J30"].apply(lambda x: pd.isna(x))
    ].index
    print(f"We found an addition of {len(index_dead_from_date)} dead patients from their date information")
    print(f"We found an addition of {len(index_alive_from_date)} living patients from their date information")
    data.loc[index_dead_from_date, "ALIVE"] = False
    data.loc[index_alive_from_date, "ALIVE"] = True
    data["DEAD"] = data["ALIVE"].apply(lambda x: 0 if (not x) is False else 1)

    data2 = data.loc[~data.loc[:, "ALIVE"].isna().values]

    print(f"There are {len(data) - len(data2)} patients with no death information at all. Ignoring them.")

    nalive = len(data2[data2["ALIVE"]])
    ndead = len(data2[data2["ALIVE"].apply(lambda x: x is False)])

    print(f"Total : {nalive} alive and {ndead} dead")
    return data2


def set_anemia(data):
    data["ANEMIE"] = pd.Series(0, index=data.index)
    idx_female = data[data["SEXE"] == "Female"].index
    idx_male = data[data["SEXE"] == "Male"].index
    anemique_female_index = data.loc[idx_female].loc[(data.loc[idx_female, "HEMOGLOBINE_PREOP"] < 12.0).values].index
    anemique_male_index = data.loc[idx_male].loc[(data.loc[idx_male, "HEMOGLOBINE_PREOP"] < 13.0).values].index
    print(f"There are {len(anemique_female_index) + len(anemique_male_index)} anaemic patients")
    data.loc[anemique_female_index, "ANEMIE"] = 1
    data.loc[anemique_male_index, "ANEMIE"] = 1
    return data


def get_fischer_df(data):
    fischer = pd.DataFrame(columns=["Alive", "Dead"], index=["Anaemic", "Not Anaemic"])
    fischer.loc["Not Anaemic", "Dead"] = len(data.loc[((data["ALIVE"] == 0) & (data["ANEMIE"] == 0)).values])
    fischer.loc["Anaemic", "Dead"] = len(data.loc[((data["ALIVE"] == 0) & (data["ANEMIE"] == 1)).values])
    fischer.loc["Anaemic", "Alive"] = len(data.loc[((data["ALIVE"] == 1) & (data["ANEMIE"] == 1)).values])
    fischer.loc["Not Anaemic", "Alive"] = len(data.loc[((data["ALIVE"] == 1) & (data["ANEMIE"] == 0)).values])
    return fischer


def fit(data):
    x_train, x_test, y_train, y_test = train_test_split(data[["HEMOGLOBINE_PREOP"]], data["DEAD"], train_size=0.9)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    score = round(model.score(x_test, y_test), 3)
    predict = model.predict_proba(data[["HEMOGLOBINE_PREOP"]].sort_values(by="HEMOGLOBINE_PREOP"))[:, 1]
    return score, predict


patients = set_anemia(set_alive(get_data()))

df = get_fischer_df(patients)
print(df)

onetail_p, twotail_p = stats.fisher_exact(df)

print(f"Fischer Exact Test : One tail P-value: {onetail_p}. Two-tail P-value {twotail_p}")

patients_male = patients[patients["SEXE"] == "Male"][["HEMOGLOBINE_PREOP", "DEAD"]]
patients_female = patients[patients["SEXE"] == "Female"][["HEMOGLOBINE_PREOP", "DEAD"]]

score_male, predict_male = fit(patients_male)
score_female, predict_female = fit(patients_female)

ax = patients_male.plot.scatter(x="HEMOGLOBINE_PREOP", y="DEAD", label="Male", c="blue")
patients_female.plot.scatter(
    x="HEMOGLOBINE_PREOP",
    y="DEAD",
    ax=ax,
    c="orange",
    label="Female",
    marker="x",
    ylabel="ALIVE                DEAD",
    xlabel="Hb (g/dL)",
)
plt.axvline(13, c="blue")
plt.axvline(12, c="orange")
plt.plot(
    patients_male["HEMOGLOBINE_PREOP"].sort_values(),
    predict_male,
    ls="--",
    c="blue",
    label=f"Male model\n(score={score_male})",
)
plt.plot(
    patients_female["HEMOGLOBINE_PREOP"].sort_values(),
    predict_female,
    ls="-.",
    c="orange",
    label=f"Female model\n(score={score_female})",
)
plt.legend(loc="right")
plt.savefig("death_vs_hb.pdf")
plt.savefig("death_vs_hb.png")
