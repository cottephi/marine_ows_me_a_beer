import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from tablewriter import TableWriter
from typing import List, Tuple, Union
from pdffactory import PdfFactory
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

"""
Define meaningful columns
"""

columns_sexe = ["SEXE"]

column_hb = ["HEMOGLOBINE_PREOP"]

columns_dates = ["DATE.INDUC", "DATE.DEATH"]

column_alive = ["ALIVE.J30"]

columns_compl = [
    "COMP.PULM",
    "COMP.CARDIAC",
    "COMP.AKI",
    "COMP.ACR",
    "COMP.PULM",
    "NSQ_Cardiac.arrest",
    "NSQ_Myocardial.infarction",
    "NSQ_Pneumonia",
    "NSQ_Pulmory.embolism",
    "NSQ_intubation",
    "NSQ_Ventilator.48h",
    "NSQ_Return.operating.room",
    "NSQ_Stroke",
    "NSQ_AKI",
    "NSQ_Deep.vein.thrombosis",
    "NSQ_Venous.thromboembolism",
    "NSQ_superf.surgical.site.infection",
    "NSQ_Deep.surgical.site.infection",
    "NSQ_Wound.disruption",
    "NSQ_Organ.space.SSI",
    "NSQ_Systemic.sepsis",
    "NSQ_Uriry.tract.infection",
]

column_transf = ["CGR.bin"]

all_columns = column_transf + column_hb + columns_compl + column_alive + columns_sexe + columns_dates

"""
Define functions
"""


def get_data() -> pd.DataFrame:
    """ Reads the data, selects the meaningful columns, drops duplicated columns, replaces OUI by 1 and NON by 0,
    drops patients with no Hb information.
    """
    data = pd.read_csv("POSE.csv", index_col=0, parse_dates=columns_dates).loc[:, all_columns].T.drop_duplicates().T
    print(f"There are {len(data)} patients")
    data = data.replace("OUI", 1)
    data = data.replace("NON", 0)
    data = data.loc[~data["HEMOGLOBINE_PREOP"].isna().values]
    print(f"There are {len(data)} patients with available Hb data")
    return data


def set_alive(data: pd.DataFrame) -> pd.DataFrame:
    """ Adds the 'ALIVE' column in the dataframe, and the 'DEAD' columns as the opposite of 'ALIVE' (no Schrödinger's
    cat allowed here).

    Uses the columns ALIVE.J30 columns to fill 'ALIVE'. This column however only indicates patients that died NOT in the
    hospital. So, next, tries to fill the missing values in ALIVE.J30 by looking for patients that have less than 30
    days between their induction date and their death date, if both are availble.

    Eventually, drops the patients with no living/death information at all.
    """
    data["ALIVE"] = pd.Series(np.nan, index=data.index)

    alive_yes = data.loc[data["ALIVE.J30"].apply(lambda x: x == 1)].index
    alive_no = data.loc[data["ALIVE.J30"].apply(lambda x: x == 0)].index

    print(f"There are {len(alive_yes)} patients that are still alive from ALIVE.J30")
    print(f"There are {len(alive_no)} patients that are dead from ALIVE.J30")

    data.loc[alive_yes, "ALIVE"] = 1
    data.loc[alive_no, "ALIVE"] = 0
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
    data.loc[index_dead_from_date, "ALIVE"] = 0
    data.loc[index_alive_from_date, "ALIVE"] = 1
    data["DEAD"] = data["ALIVE"].apply(lambda x: 0 if x == 1 else 1)

    data2 = data.loc[~data.loc[:, "ALIVE"].isna().values]

    print(f"There are {len(data) - len(data2)} patients with no death information at all. Ignoring them.")

    nalive = len(data2[data2["ALIVE"].apply(lambda x: x == 1)])
    ndead = len(data2[data2["ALIVE"].apply(lambda x: x == 0)])

    print(f"Total : {nalive} alive and {ndead} dead")
    return data2


def set_anemia(data: pd.DataFrame) -> pd.DataFrame:
    """ Adds the column 'ANEMIE' in the dataframe by looking at the value of 'HEMOGLOBINE_PREOP'. A male patient with
    HEMOGLOBINE_PREOP lower than 13 will be anaemic, a female patient with HEMOGLOBINE_PREOP lower than 12 will be
    anaemic"""
    data["ANEMIE"] = pd.Series(0, index=data.index)
    idx_female = data[data["SEXE"] == "Female"].index
    idx_male = data[data["SEXE"] == "Male"].index
    anemique_female_index = data.loc[idx_female].loc[(data.loc[idx_female, "HEMOGLOBINE_PREOP"] < 12.0).values].index
    anemique_male_index = data.loc[idx_male].loc[(data.loc[idx_male, "HEMOGLOBINE_PREOP"] < 13.0).values].index
    print(f"There are {len(anemique_female_index) + len(anemique_male_index)} anaemic patients")
    data.loc[anemique_female_index, "ANEMIE"] = 1
    data.loc[anemique_male_index, "ANEMIE"] = 1
    return data


def set_compl(data: pd.DataFrame) -> pd.DataFrame:
    """Adds the column 'COMPL' to the dataframe, which will contain 1 if any of the complication specified by
    columns_compl is 1, else 0

    Note that the value of 'ALIVE' is not at all taken into account here. A patient with 'COMPL' = 1 can be alive or
    dead, so can a patient with 'COMPL' = 0
    """
    data["COMPL"] = pd.Series(0, index=data.index)
    idx = data[data[columns_compl].any(axis=1)].index
    print(f"There are {len(idx)} patients with at least one POST-OP complication")
    data.loc[idx, "COMPL"] = 1
    return data


def set_dead_or_compl(data: pd.DataFrame) -> pd.DataFrame:
    """Adds the column 'DEAD_OR_COMPL', which contains 1 with the patient has had compliations OR is dead within
     30 days."""
    data["DEAD_OR_COMPL"] = pd.Series(0, index=data.index)
    idx = data[(data["COMPL"] == 1) | (data["ALIVE"] == 0)].index
    print(f"There are {len(idx)} patients dead or with at least one POST-OP complication")
    data.loc[idx, "DEAD_OR_COMPL"] = 1
    return data


def get_fischer_df(
    data: pd.DataFrame,
    pvals: pd.DataFrame,
    col_x: str,
    col_y: str,
    index_names: List[str],
    column_names: List[str],
    pdf_: PdfFactory,
):
    """Deas the Fischer Exact Test of col_y vs col_x, saves its pvalues in pvals, and fills the pdf object
    with the table used for the test.
    """
    fischer = pd.DataFrame(columns=column_names, index=index_names)
    fischer.loc[index_names[1], column_names[1]] = len(data.loc[((data[col_y] == 0) & (data[col_x] == 0)).values])
    fischer.loc[index_names[0], column_names[1]] = len(data.loc[((data[col_y] == 0) & (data[col_x] == 1)).values])
    fischer.loc[index_names[0], column_names[0]] = len(data.loc[((data[col_y] == 1) & (data[col_x] == 1)).values])
    fischer.loc[index_names[1], column_names[0]] = len(data.loc[((data[col_y] == 1) & (data[col_x] == 0)).values])

    print(fischer)

    pdf_.add_table(TableWriter(data=fischer))

    onetail_p, twotail_p = stats.fisher_exact(fischer)
    pvals.loc[index_names[0]] = [onetail_p, twotail_p]

    print(f"Fischer Exact Test of {col_y} vs {col_x} : One tail P-value: {onetail_p}. Two-tail P-value {twotail_p}")


def fit(data: pd.DataFrame, col_to_test: str) -> Union[Tuple[float, np.ndarray], Tuple[None, None]]:
    """From a given dataframe and a given column name, will create a train and test sample for training a
    logistic regression of col_to_test vs anaemic, then train and evaluate the model.

    Returns the score of the model along with the values of col_to_test predicted by the model.

    If there are not enough data to train the model, returns None and None.
    """
    x_train, x_test, y_train, y_test = train_test_split(data[["HEMOGLOBINE_PREOP"]], data[col_to_test], train_size=0.9)
    if np.count_nonzero(y_train.values) < 2:
        print(f"Not enough data to make a logistic regression on {col_to_test}")
        return None, None

    model = LogisticRegression()
    model.fit(x_train, y_train)
    score = round(model.score(x_test, y_test), 3)
    predict = model.predict_proba(data[["HEMOGLOBINE_PREOP"]].sort_values(by="HEMOGLOBINE_PREOP"))[:, 1]
    return score, predict


def make_logistic(df: pd.DataFrame, col_to_test: str, pdf_: PdfFactory):
    """Will call 'fit' with the given dataframe, col_to_test and pdf. Will plot the fit results and add the figure
    to the pdf."""

    patients_male = df[patients["SEXE"] == "Male"][["HEMOGLOBINE_PREOP", col_to_test]]
    patients_female = df[patients["SEXE"] == "Female"][["HEMOGLOBINE_PREOP", col_to_test]]

    score_male, predict_male = fit(patients_male, col_to_test)
    score_female, predict_female = fit(patients_female, col_to_test)

    if score_male is None or score_female is None:
        return

    ax = patients_male.plot.scatter(x="HEMOGLOBINE_PREOP", y=col_to_test, label="Male", c="blue")
    patients_female.plot.scatter(
        x="HEMOGLOBINE_PREOP",
        y=col_to_test,
        ax=ax,
        c="orange",
        label="Female",
        marker="x",
        ylabel="ok                not ok",
        xlabel="Hb (g/dL)",
        title=f"{col_to_test} vs Hb",
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
    pdf_.add_figure(plt.gcf())
    plt.close()


def format_x(x):
    if x > 0.01:
        return str(round(x, 2))
    xstr = "{:.4E}".format(x)
    lead, tail = xstr.split("E-")
    while tail.startswith("0"):
        tail = tail[1:]
    xstr = "$\\times 10^{-".join([lead, tail]) + "}$"
    if x < 0.05:
        xstr = "\\textcolor{Green}{" + xstr + "}"
    return xstr


patients = set_dead_or_compl(set_compl(set_anemia(set_alive(get_data()))))

df_pvalues = pd.DataFrame(columns=["One-tail P-value", "Two-tail P-value"])
df_pvalues.index.name = "vs Anemia"
pdf_anemia = PdfFactory("tables_hb.pdf")
pdf_transfu = PdfFactory("tables_transfu.pdf")

"""
Fisher Exact Tests
"""

# Dead vs Anemia
get_fischer_df(patients, df_pvalues, "ANEMIE", "DEAD", ["Anaemic", "Not Anaemic"], ["Dead", "Alive"], pdf_anemia)
# Dead vs Transfusion
get_fischer_df(
    patients, df_pvalues, column_transf[0], "DEAD", ["Transfusion", "No Transfusion"], ["Dead", "Alive"], pdf_transfu
)

# Complications vs Anemia
get_fischer_df(
    patients,
    df_pvalues,
    "ANEMIE",
    "COMPL",
    ["Anaemic", "Not Anaemic"],
    ["Complications", "No complications"],
    pdf_anemia,
)
# Complications vs Transfusion
get_fischer_df(
    patients,
    df_pvalues,
    column_transf[0],
    "COMPL",
    ["Transfusion", "No Transfusion"],
    ["Complications", "No complications"],
    pdf_transfu,
)

# Dead or Complications vs Anemia
get_fischer_df(
    patients,
    df_pvalues,
    "ANEMIE",
    "DEAD_OR_COMPL",
    ["Anaemic", "Not Anaemic"],
    ["Complications or death", "Alive and well"],
    pdf_anemia,
)
# Dead or Complications vs Transfusion
get_fischer_df(
    patients,
    df_pvalues,
    column_transf[0],
    "DEAD_OR_COMPL",
    ["Transfusion", "No Transfusion"],
    ["Complications or death", "Alive and well"],
    pdf_transfu,
)

for compl in columns_compl:
    # One complication vs Anemia
    get_fischer_df(
        patients,
        df_pvalues,
        "ANEMIE",
        compl,
        ["Anaemic", "Not Anaemic"],
        [compl, f"No {compl}"],
        pdf_anemia
    )
    # One complication vs Transfusion
    get_fischer_df(
        patients,
        df_pvalues,
        column_transf[0],
        compl,
        ["Transfusion", "No Transfusion"],
        [compl, f"No {compl}"],
        pdf_transfu,
    )

df_pvalues = df_pvalues.applymap(format_x).astype(str)
writer = TableWriter("table.pdf", data=df_pvalues)

pdf_anemia.add_table(writer)

"""
Logistic Regressions
"""

# Alive vs Anemia
make_logistic(patients, "DEAD", pdf_anemia)

# Complications vs Anemia
make_logistic(patients, "COMPL", pdf_anemia)

# Dead or Complications vs Anemia
make_logistic(patients, "DEAD_OR_COMPL", pdf_anemia)

# One complication vs Anemia
for compl in columns_compl:
    make_logistic(patients, compl, pdf_anemia)
