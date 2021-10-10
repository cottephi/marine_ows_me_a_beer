from typing import List, Union, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pdffactory import PdfFactory
from scipy import stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tablewriter import TableWriter


def get_data() -> pd.DataFrame:
    """ Reads the data, selects the meaningful columns, drops duplicated columns, replaces OUI by 1 and NON by 0,
    drops patients with no Hb information.
    """
    data = pd.read_csv("POSE.csv", index_col=0, parse_dates=columns_dates).loc[:, all_columns]
    data_fragilites = pd.read_csv("fragilites.csv", index_col=0)
    data = pd.concat([data, data_fragilites], axis=1).T.drop_duplicates().T
    print(f"There are {len(data)} patients")
    data = data.replace("OUI", 1)
    data = data.replace("NON", 0)
    data = data.loc[~data[column_hb[0]].isna().values]
    print(f"There are {len(data)} patients with available Hb data")
    return data


def set_alive(data: pd.DataFrame):
    """ Adds the 'ALIVE' column in the dataframe, and the 'DEAD' columns as the opposite of 'ALIVE' (no Schrödinger's
    cat allowed here).

    Uses the columns ALIVE.J30 columns to fill 'ALIVE'. This column however only indicates patients that died NOT in the
    hospital. So, next, tries to fill the missing values in ALIVE.J30 by looking for patients that have less than 30
    days between their induction date and their death date, if both are availble.

    Eventually, drops the patients with no living/death information at all.
    """
    data.loc[:, "ALIVE"] = pd.Series(np.nan, index=data.index)

    alive_yes = data.loc[data[column_alive[0]].apply(lambda x: x == 1)].index
    alive_no = data.loc[data[column_alive[0]].apply(lambda x: x == 0)].index

    print(f"There are {len(alive_yes)} patients that are still alive from ALIVE.J30")
    print(f"There are {len(alive_no)} patients that are dead from ALIVE.J30")

    data.loc[alive_yes, "ALIVE"] = 1
    data.loc[alive_no, "ALIVE"] = 0
    index_dead_from_date = data[(data[columns_dates[1]] - data[columns_dates[0]]).apply(lambda x: x.days < 30)].index

    index_dead_from_date = data.loc[index_dead_from_date][
        data.loc[index_dead_from_date, column_alive[0]].apply(lambda x: pd.isna(x))
    ].index
    index_alive_from_date = data[(data[columns_dates[1]] - data[columns_dates[0]]).apply(lambda x: x.days >= 30)].index
    index_alive_from_date = data.loc[index_alive_from_date][
        data.loc[index_alive_from_date, column_alive[0]].apply(lambda x: pd.isna(x))
    ].index
    print(f"We found an addition of {len(index_dead_from_date)} dead patients from their date information")
    print(f"We found an addition of {len(index_alive_from_date)} living patients from their date information")
    data.loc[index_dead_from_date, "ALIVE"] = 0
    data.loc[index_alive_from_date, "ALIVE"] = 1
    data["DEAD"] = data["ALIVE"].apply(lambda x: 0 if x == 1 else 1)

    idx = data.loc[data.loc[:, "ALIVE"].isna().values].index

    print(f"There are {len(idx)} patients with no death information at all. Ignoring them.")
    data.drop(idx, inplace=True)

    nalive = len(data[data["ALIVE"].apply(lambda x: x == 1)])
    ndead = len(data[data["ALIVE"].apply(lambda x: x == 0)])

    print(f"Total : {nalive} alive and {ndead} dead")


def set_time_of_death(data: pd.DataFrame):
    """Will add the column 'DEAD_DAYS', which is the number of days between the patient induction and death."""
    data.loc[:, "DEAD_DAYS"] = (data[columns_dates[1]] - data[columns_dates[0]]).apply(lambda x: x.days)


def set_anemia(data: pd.DataFrame):
    """ Adds the column 'ANEMIE' in the dataframe by looking at the value of 'HEMOGLOBINE_PREOP'. A male patient with
    HEMOGLOBINE_PREOP lower than 13 will be anaemic, a female patient with HEMOGLOBINE_PREOP lower than 12 will be
    anaemic"""
    data.loc[:, "ANEMIE"] = pd.Series(0, index=data.index)
    idx_female = data[data[columns_sexe[0]] == "Female"].index
    idx_male = data[data[columns_sexe[0]] == "Male"].index
    anemique_female_index = data.loc[idx_female].loc[(data.loc[idx_female, column_hb[0]] < 12.0).values].index
    anemique_male_index = data.loc[idx_male].loc[(data.loc[idx_male, column_hb[0]] < 13.0).values].index
    print(f"There are {len(anemique_female_index) + len(anemique_male_index)} anaemic patients")
    data.loc[anemique_female_index, "ANEMIE"] = 1
    data.loc[anemique_male_index, "ANEMIE"] = 1


def set_compl(data: pd.DataFrame):
    """Adds the column 'COMPL' to the dataframe, which will contain 1 if any of the complication specified by
    columns_compl is 1, else 0

    Note that the value of 'ALIVE' is not at all taken into account here. A patient with 'COMPL' = 1 can be alive or
    dead, so can a patient with 'COMPL' = 0
    """
    data.loc[:, "COMPL"] = pd.Series(0, index=data.index)
    idx = data[data[columns_compl].any(axis=1)].index
    print(f"There are {len(idx)} patients with at least one POST-OP complication")
    data.loc[idx, "COMPL"] = 1


def set_dead_or_compl(data: pd.DataFrame):
    """Adds the column 'DEAD_OR_COMPL', which contains 1 with the patient has had compliations OR is dead within
     30 days."""
    data.loc[:, "DEAD_OR_COMPL"] = pd.Series(0, index=data.index)
    idx = data[(data["COMPL"] == 1) | (data["ALIVE"] == 0)].index
    print(f"There are {len(idx)} patients dead or with at least one POST-OP complication")
    data.loc[idx, "DEAD_OR_COMPL"] = 1


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
    pvals.loc[column_names[0]] = [onetail_p, twotail_p]

    print(f"Fischer Exact Test of {col_y} vs {col_x} : One tail P-value: {onetail_p}. Two-tail P-value {twotail_p}")


def fit(data: pd.DataFrame, col_to_test: str) -> Union[Tuple[float, np.ndarray], Tuple[None, None]]:
    """From a given dataframe and a given column name, will create a train and test sample for training a
    logistic regression of col_to_test vs anaemic, then train and evaluate the model.

    Returns the score of the model along with the values of col_to_test predicted by the model.

    If there are not enough data to train the model, returns None and None.
    """
    x_train, x_test, y_train, y_test = train_test_split(data[[column_hb[0]]], data[col_to_test], train_size=0.9)
    if np.count_nonzero(y_train.values) < 2:
        print(f"Not enough data to make a logistic regression on {col_to_test}")
        return None, None

    model = LogisticRegression()
    model.fit(x_train, y_train)
    score = round(model.score(x_test, y_test), 3)
    predict = model.predict_proba(data[[column_hb[0]]].sort_values(by=column_hb[0]))[:, 1]
    return score, predict


def make_logistic(df: pd.DataFrame, col_to_test: str, pdf_: PdfFactory):
    """Will call 'fit' with the given dataframe, col_to_test and pdf. Will plot the fit results and add the figure
    to the pdf."""

    patients_male = df[df[columns_sexe[0]] == "Male"][[column_hb[0], col_to_test]]
    patients_female = df[df[columns_sexe[0]] == "Female"][[column_hb[0], col_to_test]]

    score_male, predict_male = fit(patients_male, col_to_test)
    score_female, predict_female = fit(patients_female, col_to_test)

    if score_male is None or score_female is None:
        return

    ax = patients_male.plot.scatter(x=column_hb[0], y=col_to_test, label="Male", c="blue")
    patients_female.plot.scatter(
        x=column_hb[0],
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
        patients_male[column_hb[0]].sort_values(),
        predict_male,
        ls="--",
        c="blue",
        label=f"Male model\n(score={score_male})",
    )
    plt.plot(
        patients_female[column_hb[0]].sort_values(),
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
    if "E-" in xstr:
        lead, tail = xstr.split("E-")
        middle = "-"
    else:
        lead, tail = xstr.split("E")
        middle = ""
    while tail.startswith("0"):
        tail = tail[1:]
    xstr = ("$\\times 10^{" + middle).join([lead, tail]) + "}$"
    if x < alpha:
        xstr = "\\textcolor{Green}{" + xstr + "}"
    return xstr


alpha = 0.05  # P-value limit

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
