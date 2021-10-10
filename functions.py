import warnings
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pdffactory import PdfFactory
from scipy import stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tablewriter import TableWriter
import lifelines


def printline(m):
    print(m, "\n")


def get_data() -> pd.DataFrame:
    """ Reads the data, selects the meaningful columns, drops duplicated columns, replaces OUI by 1 and NON by 0
    """
    data = pd.read_csv("POSE.csv", index_col=0, parse_dates=columns_dates)
    data_fragilites = pd.read_csv("fragilites.csv", index_col=0)
    data = pd.concat([data, data_fragilites], axis=1).T.drop_duplicates().T.loc[:, all_columns]
    printline(f"There are {len(data)} patients")
    data = data.replace("OUI", 1)
    data = data.replace("NON", 0)
    return data


def set_binned_age(data: pd.DataFrame):
    """Will add the column 'BINNED_AGE' in the dataframe, discretizing the patients'age in bins [0-85[, [85-90[
     and > 90, np.nan if unknown """

    def f(x: int):
        if np.isnan(x):
            return np.nan
        if x < 85:
            return 0
        if x < 90:
            return 1
        return 2

    data.loc[:, "BINNED_AGE"] = data.loc[:, column_age[0]].apply(f)
    printline(f"There are {len(data[data[column_age[0]].isna()])} patients with no age information")


def set_binned_sexe(data: pd.DataFrame):
    """Will add the column 'BINNED_AGE' in the dataframe, discretizing the patients'age in bins [0-85[, [85-90[
     and > 90, np.nan if unknown """

    def f(x: str):
        if not isinstance(x, str):
            if np.isnan(x):
                return np.nan
            else:
                raise ValueError(x)
        if x == "Male":
            return 1
        if x == "Female":
            return 0
        raise ValueError(x)

    data.loc[:, "BINNED_SEXE"] = data.loc[:, column_sexe[0]].apply(f)
    printline(f"There are {len(data[data[column_age[0]].isna()])} patients with no age information")


def set_ag(data: pd.DataFrame):
    """Will add the column 'AG', containing 1 if the anaesthesis was general, else 0, np.nan if unknown"""

    def f(x: str):
        if not isinstance(x, str):
            if np.isnan(x):
                return np.nan
            else:
                raise ValueError(x)
        if x == "AG":
            return 1
        return 0

    data.loc[:, "AG"] = data.loc[:, column_ag[0]].apply(f)
    printline(f"There are {len(data[data[column_ag[0]].isna()])} patients with no General Anaesthesis information")


def set_severity(data: pd.DataFrame):
    """Will add the column 'SEVERITY', containing the severity score of the surgery, or np.nan if unknown"""

    def f(x: str):
        if not isinstance(x, str):
            if np.isnan(x):
                return np.nan
            else:
                raise ValueError(x)
        if x == "MAJOR":
            return 2
        if x == "MINOR":
            return 1
        return 0

    data.loc[:, "SEVERITY"] = data.loc[:, column_sev[0]].apply(f)
    printline(f"There are {len(data[data[column_sev[0]].isna()])} patients with no chir. severity information")


def set_morbidite(data: pd.DataFrame):
    """Will add the column 'MORBIDITE', containing 1 if the patient has at least 2 morbidity factors, 1 if not,
    and np.nan if unknown"""
    def f(x: int):
        if np.isnan(x):
            return np.nan
        if x >= 4:
            return 1
        return 0

    data.loc[:, "MORBIDITE"] = data.loc[:, column_comor[0]].apply(f)
    printline(f"There are {len(data[data[column_comor[0]].isna()])} patients no with comorbisity information")


def set_fragility(data: pd.DataFrame):
    """Will add the column 'FRAGILE', containing 1 if the patient has more than 4 fragilities, 1 if not,
    and np.nan if unknown"""

    def f(x: int):
        if np.isnan(x):
            return np.nan
        if x >= 4:
            return 1
        return 0

    data.loc[:, "FRAGILE"] = data.loc[:, column_fragility[0]].apply(f)
    printline(f"There are {len(data[data[column_fragility[0]].isna()])} patients no with fragility information")


def set_prog(data: pd.DataFrame):
    """Will add the column 'URGENTE', containing 1 if the surgery was programmed, else 0, np.nan if unknown"""

    def f(x: str):
        if not isinstance(x, str):
            if np.isnan(x):
                return np.nan
            else:
                raise ValueError(x)
        if x == "URGENT":
            return 1
        return 0

    data.loc[:, "URGENTE"] = data.loc[:, column_prog[0]].apply(f)
    printline(
        f"There are {len(data[data[column_prog[0]].isna()])} patients with no information about operation programmation"
    )


def set_time_of_death(data: pd.DataFrame):
    """Will add the column 'DEAD_DAYS', which is the number of days between the patient induction and death."""
    data.loc[:, "DEAD_DAYS"] = (data[columns_dates[1]] - data[columns_dates[0]]).apply(lambda x: x.days)
    print(
        f"There are {len(data[~data[columns_dates[0]].isna() & ~data[columns_dates[1]].isna()])} patients with"
        " death date information"
    )
    printline(
        f"There are {len(data[data[columns_dates[0]].isna() | data[columns_dates[1]].isna()])} patients with no"
        " death date information"
    )


def set_alive(data: pd.DataFrame):
    """ Adds the 'ALIVE' column in the dataframe, and the 'DEAD' columns as the opposite of 'ALIVE' (no Schrödinger's
    cat allowed here).

    Uses the columns ALIVE.J30 columns to fill 'ALIVE'. This column however only indicates patients that died NOT in the
    hospital. So, next, tries to fill the missing values in ALIVE.J30 by looking for patients that have less than 30
    days between their induction date and their death date, if both are availble.

    If no information is available on life and death, the column contains np.nan
    """
    data.loc[:, "ALIVE"] = pd.Series(np.nan, index=data.index)

    alive_yes = data.loc[data[column_alive[0]].apply(lambda x: x == 1)].index
    alive_no = data.loc[data[column_alive[0]].apply(lambda x: x == 0)].index

    print(f"There are {len(alive_yes)} patients that are still alive from ALIVE.J30")
    print(f"There are {len(alive_no)} patients that are dead from ALIVE.J30")
    printline(f"There are {len(data[data[column_alive[0]].isna()])} patients with no ALIVE.J30 information")

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
    printline(f"We found an addition of {len(index_alive_from_date)} living patients from their date information")
    data.loc[index_dead_from_date, "ALIVE"] = 0
    data.loc[index_alive_from_date, "ALIVE"] = 1

    idx = data.loc[data.loc[:, "ALIVE"].isna().values].index
    data.loc[idx, "ALIVE"] = np.nan

    def f(x):
        if np.isnan(x):
            return np.nan
        else:
            return int(not x)
    data["DEAD"] = data["ALIVE"].apply(f)

    nalive = len(data[data["ALIVE"].apply(lambda x: x == 1)])
    ndead = len(data[data["ALIVE"].apply(lambda x: x == 0)])

    printline(f"Total : {nalive} alive, {ndead} dead, {len(data) - (nalive + ndead)} without information")


def set_anemia(data: pd.DataFrame):
    """ Adds the column 'ANEMIE' in the dataframe by looking at the value of 'HEMOGLOBINE_PREOP'. A male patient with
    HEMOGLOBINE_PREOP lower than 13 will be anaemic, a female patient with HEMOGLOBINE_PREOP lower than 12 will be
    anaemic"""
    data.loc[:, "ANEMIE"] = pd.Series(np.nan, index=data.index)
    idx_female = data[data[column_sexe[0]] == "Female"].index
    idx_male = data[data[column_sexe[0]] == "Male"].index
    anemique_female_index = data.loc[idx_female].loc[(data.loc[idx_female, column_hb[0]] < 12.0).values].index
    not_anemique_female_index = data.loc[idx_female].loc[(data.loc[idx_female, column_hb[0]] >= 12.0).values].index
    anemique_male_index = data.loc[idx_male].loc[(data.loc[idx_male, column_hb[0]] < 13.0).values].index
    not_anemique_male_index = data.loc[idx_male].loc[(data.loc[idx_male, column_hb[0]] >= 13.0).values].index
    data.loc[anemique_female_index, "ANEMIE"] = 1
    data.loc[anemique_male_index, "ANEMIE"] = 1
    data.loc[not_anemique_female_index, "ANEMIE"] = 0
    data.loc[not_anemique_male_index, "ANEMIE"] = 0

    nmissing = len(data[data["ANEMIE"].isna()])
    assert nmissing + len(anemique_female_index) + len(anemique_male_index) + len(not_anemique_female_index) + len(
        not_anemique_male_index
    ) == len(data)
    print(f"There are {len(anemique_female_index) + len(anemique_male_index)} anaemic patients")
    print(f"There are {len(not_anemique_female_index) + len(not_anemique_male_index)} not anaemic patients")
    printline(f"There are {nmissing} patients without Hb informations")


def set_compl(data: pd.DataFrame):
    """Adds the column 'COMPL' to the dataframe, which will contain 1 if any of the complication specified by
    columns_compl is 1, 0 if no complications where observed, np.nan impossible to know,

    Note that the value of 'ALIVE' is not at all taken into account here. A patient with 'COMPL' = 1 can be alive or
    dead, so can a patient with 'COMPL' = 0
    """
    data.loc[:, "COMPL"] = pd.Series(0, index=data.index)
    idx = data[data[columns_compl].any(axis=1)].index
    idx_nans = data[(data[columns_compl].isna().any(axis=1)) & ~data[columns_compl].any(axis=1)].index
    print(f"There are {len(idx)} patients with at least one POST-OP complication")
    printline(f"There are {len(idx_nans)} patients with no useable information about POST-OP complication")
    data.loc[idx, "COMPL"] = 1
    data.loc[idx_nans, "COMPL"] = np.nan


def set_dead_or_compl(data: pd.DataFrame):
    """Adds the column 'DEAD_OR_COMPL', which contains 1 with the patient has had compliations OR is dead within
     30 days."""
    data.loc[:, "DEAD_OR_COMPL"] = pd.Series(np.nan, index=data.index)
    idx = data[(data["COMPL"] == 1) | (data["ALIVE"] == 0)].index
    print(f"There are {len(idx)} patients dead or with at least one POST-OP complication")
    data.loc[idx, "DEAD_OR_COMPL"] = 1
    idx = data[(data["COMPL"] == 0) & (data["ALIVE"] == 1)].index
    printline(f"There are {len(idx)} patients alive and well")
    data.loc[idx, "DEAD_OR_COMPL"] = 0


def set_censure(data: pd.DataFrame):
    """Will create a new column CENSURE indicating if the death date information is available (1) or not (0)"""
    data.loc[:, "CENSURE"] = data.loc[:, "DEAD_DAYS"].apply(lambda x: 0 if np.isnan(x) else 1)


def drop_for_cox(data: pd.DataFrame):
    """Will remove patients that have at least one missing explaining feature for COX regression"""
    to_drop = data[
        data["BINNED_AGE"].isna()
        | data["FRAGILE"].isna()
        | data["SEVERITY"].isna()
        | data["URGENTE"].isna()
        | data["IRA"].isna()
        | data["AG"].isna()
    ].index
    data.drop(to_drop, inplace=True)
    printline(f"There were {len(to_drop)} patients with at least one missing explaining feature")


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

    printline(f"Fischer Exact Test of {col_y} vs {col_x} : One tail P-value: {onetail_p}. Two-tail P-value {twotail_p}")


def fit(data: pd.DataFrame, col_to_test: str) -> Union[Tuple[float, np.ndarray], Tuple[None, None]]:
    """From a given dataframe and a given column name, will create a train and test sample for training a
    logistic regression of col_to_test vs anaemic, then train and evaluate the model.

    Returns the score of the model along with the values of col_to_test predicted by the model.

    If there are not enough data to train the model, returns None and None.
    """
    x_train, x_test, y_train, y_test = train_test_split(data[[column_hb[0]]], data[col_to_test], train_size=0.9)
    if np.count_nonzero(y_train.values) < 2:
        warnings.warn(f"Not enough data to make a logistic regression on {col_to_test}")
        return None, None

    model = LogisticRegression()
    model.fit(x_train, y_train)
    score = round(model.score(x_test, y_test), 3)
    predict = model.predict_proba(data[[column_hb[0]]].sort_values(by=column_hb[0]))[:, 1]
    return score, predict


def make_logistic(df: pd.DataFrame, col_to_test: str, pdf_: PdfFactory):
    """Will call 'fit' with the given dataframe, col_to_test and pdf. Will plot the fit results and add the figure
    to the pdf."""

    df = df[~df[col_to_test].isna() & ~df[column_hb[0]].isna()]

    patients_male = df[df[column_sexe[0]] == "Male"][[column_hb[0], col_to_test]]
    patients_female = df[df[column_sexe[0]] == "Female"][[column_hb[0], col_to_test]]

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
    """For a given number, will put it in scientific notation if lower than 0.01, using LaTex synthax.

    In addition, if the value is lower than alpha, will change set the color of the value to green.
    """
    if x > 0.01:
        xstr = str(round(x, 2))
    else:
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


def init_cox():
    df = pd.read_csv("cox_empty.csv", header=None)
    df = df.set_index([df.columns[0], df.columns[1]])
    df.columns = pd.MultiIndex.from_arrays([df.iloc[0], df.iloc[1]])
    df = df.iloc[1:]
    idx = list(df.index)
    idx[0] = ("", "Total")
    df.index = pd.MultiIndex.from_tuples(idx)
    df.iloc[0] = np.nan
    df.index.names = ["", ""]
    df.columns.names = ["", ""]
    return df


def fill_cox(data):

    df_cox = init_cox()

    """ Age """

    # Dead anaemic
    df_cox.loc[("", "Total"), ("Mortalité J30", "Anémie")] = len(data[(data["ANEMIE"] == 1) & (data["DEAD"] == 1)])

    df_cox.loc[("Age", "80-84"), ("Mortalité J30", "Anémie")] = len(
        data[(data["BINNED_AGE"] == 0) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Age", "85-89"), ("Mortalité J30", "Anémie")] = len(
        data[(data["BINNED_AGE"] == 1) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Age", ">90"), ("Mortalité J30", "Anémie")] = len(
        data[(data["BINNED_AGE"] == 2) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )

    # Dead not anaemic
    df_cox.loc[("", "Total"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )

    df_cox.loc[("Age", "80-84"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["BINNED_AGE"] == 0) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Age", "85-89"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["BINNED_AGE"] == 1) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Age", ">90"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["BINNED_AGE"] == 2) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )

    # Complications or dead anaemic
    df_cox.loc[("", "Total"), ("Complications J30", "Anémie")] = len(data[(data["ANEMIE"] == 1) & (data["COMPL"] == 1)])

    df_cox.loc[("Age", "80-84"), ("Complications J30", "Anémie")] = len(
        data[(data["BINNED_AGE"] == 0) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Age", "85-89"), ("Complications J30", "Anémie")] = len(
        data[(data["BINNED_AGE"] == 1) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Age", ">90"), ("Complications J30", "Anémie")] = len(
        data[(data["BINNED_AGE"] == 2) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )

    # Complications or dead not anaemic
    df_cox.loc[("", "Total"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )

    df_cox.loc[("Age", "80-84"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["BINNED_AGE"] == 0) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Age", "85-89"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["BINNED_AGE"] == 1) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Age", ">90"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["BINNED_AGE"] == 2) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )

    """ AG """

    # Dead anaemic
    df_cox.loc[("Anesthesie", "AG"), ("Mortalité J30", "Anémie")] = len(
        data[(data["AG"] == 1) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Anesthesie", "Autre"), ("Mortalité J30", "Anémie")] = len(
        data[(data["AG"] == 0) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )

    # Dead not anaemic
    df_cox.loc[("Anesthesie", "AG"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["AG"] == 1) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Anesthesie", "Autre"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["AG"] == 0) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )

    # Complications or dead anaemic
    df_cox.loc[("Anesthesie", "AG"), ("Complications J30", "Anémie")] = len(
        data[(data["AG"] == 1) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Anesthesie", "Autre"), ("Complications J30", "Anémie")] = len(
        data[(data["AG"] == 0) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )

    # Complications or dead not anaemic
    df_cox.loc[("Anesthesie", "AG"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["AG"] == 1) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Anesthesie", "Autre"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["AG"] == 0) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )

    """ Prog """

    # Dead anaemic
    df_cox.loc[("Type chirurgie", "Urgente"), ("Mortalité J30", "Anémie")] = len(
        data[(data["URGENTE"] == 1) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Type chirurgie", "Programmee"), ("Mortalité J30", "Anémie")] = len(
        data[(data["URGENTE"] == 0) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )

    # Dead not anaemic
    df_cox.loc[("Type chirurgie", "Urgente"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["URGENTE"] == 1) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Type chirurgie", "Programmee"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["URGENTE"] == 0) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )

    # Complications or dead anaemic
    df_cox.loc[("Type chirurgie", "Urgente"), ("Complications J30", "Anémie")] = len(
        data[(data["URGENTE"] == 1) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Type chirurgie", "Programmee"), ("Complications J30", "Anémie")] = len(
        data[(data["URGENTE"] == 0) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )

    # Complications or dead not anaemic
    df_cox.loc[("Type chirurgie", "Urgente"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["URGENTE"] == 1) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Type chirurgie", "Programmee"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["URGENTE"] == 0) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )

    """ Severity """

    # Dead anaemic
    df_cox.loc[("Severite chirurgie", "Majeure"), ("Mortalité J30", "Anémie")] = len(
        data[(data["SEVERITY"] == 2) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Severite chirurgie", "Intermediaire"), ("Mortalité J30", "Anémie")] = len(
        data[(data["SEVERITY"] == 1) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Severite chirurgie", "Mineure"), ("Mortalité J30", "Anémie")] = len(
        data[(data["SEVERITY"] == 0) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )

    # Dead not anaemic
    df_cox.loc[("Severite chirurgie", "Majeure"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["SEVERITY"] == 2) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Severite chirurgie", "Intermediaire"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["SEVERITY"] == 1) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Severite chirurgie", "Mineure"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["SEVERITY"] == 0) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )

    # Complications or dead anaemic
    df_cox.loc[("Severite chirurgie", "Majeure"), ("Complications J30", "Anémie")] = len(
        data[(data["SEVERITY"] == 2) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Severite chirurgie", "Intermediaire"), ("Complications J30", "Anémie")] = len(
        data[(data["SEVERITY"] == 1) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Severite chirurgie", "Mineure"), ("Complications J30", "Anémie")] = len(
        data[(data["SEVERITY"] == 0) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )

    # Complications or dead not anaemic
    df_cox.loc[("Severite chirurgie", "Majeure"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["SEVERITY"] == 2) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Severite chirurgie", "Intermediaire"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["SEVERITY"] == 1) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Severite chirurgie", "Mineure"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["SEVERITY"] == 0) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )

    """ Fragility """

    # Dead anaemic
    df_cox.loc[("Fragilite", "Presence"), ("Mortalité J30", "Anémie")] = len(
        data[(data["FRAGILE"] == 1) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Fragilite", "Absence"), ("Mortalité J30", "Anémie")] = len(
        data[(data["FRAGILE"] == 0) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )

    # Dead not anaemic
    df_cox.loc[("Fragilite", "Presence"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["FRAGILE"] == 1) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("Fragilite", "Absence"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["FRAGILE"] == 0) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )

    # Complications or dead anaemic
    df_cox.loc[("Fragilite", "Presence"), ("Complications J30", "Anémie")] = len(
        data[(data["FRAGILE"] == 1) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Fragilite", "Absence"), ("Complications J30", "Anémie")] = len(
        data[(data["FRAGILE"] == 0) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )

    # Complications or dead not anaemic
    df_cox.loc[("Fragilite", "Presence"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["FRAGILE"] == 1) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("Fragilite", "Absence"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["FRAGILE"] == 0) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )

    """ IRA """

    # Dead anaemic
    df_cox.loc[("IRA", "Presence"), ("Mortalité J30", "Anémie")] = len(
        data[(data["IRA"] == 1) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("IRA", "Absence"), ("Mortalité J30", "Anémie")] = len(
        data[(data["IRA"] == 0) & (data["ANEMIE"] == 1) & (data["DEAD"] == 1)]
    )

    # Dead not anaemic
    df_cox.loc[("IRA", "Presence"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["IRA"] == 1) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )
    df_cox.loc[("IRA", "Absence"), ("Mortalité J30", "Pas d'anémie")] = len(
        data[(data["IRA"] == 0) & (data["ANEMIE"] == 0) & (data["DEAD"] == 1)]
    )

    # Complications or dead anaemic
    df_cox.loc[("IRA", "Presence"), ("Complications J30", "Anémie")] = len(
        data[(data["IRA"] == 1) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("IRA", "Absence"), ("Complications J30", "Anémie")] = len(
        data[(data["IRA"] == 0) & (data["ANEMIE"] == 1) & (data["COMPL"] == 1)]
    )

    # Complications or dead not anaemic
    df_cox.loc[("IRA", "Presence"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["IRA"] == 1) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )
    df_cox.loc[("IRA", "Absence"), ("Complications J30", "Pas d'anémie")] = len(
        data[(data["IRA"] == 0) & (data["ANEMIE"] == 0) & (data["COMPL"] == 1)]
    )

    return df_cox


def fit_cox(data: pd.DataFrame):
    printline(f"There are {len(data)} patients in the COX dataset")
    data = data[columns_cox]
    data.loc[:, "DEAD_DAYS"] = data.loc[:, "DEAD_DAYS"].fillna(30)
    cpf = lifelines.CoxPHFitter()
    return cpf.fit(data, duration_col="DEAD_DAYS", event_col="CENSURE")


alpha = 0.05  # P-value limit

column_sexe = ["SEXE"]
column_age = ["AGE"]
column_prog = ["Surgery"]
column_sev = ["Severity.surgery"]
column_fragility = ["fragilite"]
column_ag = ["ANESTH.TECH"]
column_ira = ["IRA"]
column_comor = ["COMORBIDITES"]
column_hb = ["HEMOGLOBINE_PREOP"]
columns_dates = ["DATE.INDUC", "DATE.DEATH"]
column_alive = ["ALIVE.J30"]
column_transf = ["CGR.bin"]
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
    "DEVENIR.STROKE"
]
columns_cox = [
    "BINNED_AGE",
    "DEAD_DAYS",
    "URGENTE",
    "FRAGILE",
    "SEVERITY",
    # "IRA",
    "AG",
    "CENSURE",
    "CGR.bin",
    "MORBIDITE",
    "BINNED_SEXE"
]

all_columns = (
        column_transf
        + column_hb
        + columns_compl
        + column_alive
        + column_sexe
        + columns_dates
        + column_age
        + column_ira
        + column_ag
        + column_sev
        + column_prog
        + column_comor
        + column_fragility
)
