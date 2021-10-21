import pandas as pd
from tablewriter import TableWriter
from pdffactory import PdfFactory

from functions import (
    get_data,
    set_alive,
    set_anemia,
    set_compl,
    set_fragility,
    set_dead_or_compl,
    get_fischer_df,
    make_logistic,
    format_x, columns_compl, column_transf,
)

patients = get_data()
set_alive(patients)
set_anemia(patients)
set_compl(patients)
set_fragility(patients)
set_dead_or_compl(patients)

patients = patients.T.drop_duplicates().T

df_pvalues_anemia = pd.DataFrame(columns=["One-tail P-value", "Two-tail P-value"])
df_pvalues_anemia.index.name = "vs Anemia"
df_pvalues_transfu = pd.DataFrame(columns=["One-tail P-value", "Two-tail P-value"])
df_pvalues_transfu.index.name = "vs Transfusion"
pdf_anemia = PdfFactory("tables_hb.pdf")
pdf_transfu = PdfFactory("tables_transfu.pdf")

"""
Fisher Exact Tests
"""

# Dead vs Anemia
get_fischer_df(patients, df_pvalues_anemia, "ANEMIE", "DEAD", ["Anaemic", "Not Anaemic"], ["Dead", "Alive"], pdf_anemia)
# Dead vs Transfusion
get_fischer_df(
    patients,
    df_pvalues_transfu,
    column_transf[0],
    "DEAD",
    ["Transfusion", "No Transfusion"],
    ["Dead", "Alive"],
    pdf_transfu,
)

# Complications vs Anemia
get_fischer_df(
    patients,
    df_pvalues_anemia,
    "ANEMIE",
    "COMPL",
    ["Anaemic", "Not Anaemic"],
    ["Complications", "No complications"],
    pdf_anemia,
)
# Complications vs Transfusion
get_fischer_df(
    patients,
    df_pvalues_transfu,
    column_transf[0],
    "COMPL",
    ["Transfusion", "No Transfusion"],
    ["Complications", "No complications"],
    pdf_transfu,
)

# Dead or Complications vs Anemia
get_fischer_df(
    patients,
    df_pvalues_anemia,
    "ANEMIE",
    "DEAD_OR_COMPL",
    ["Anaemic", "Not Anaemic"],
    ["Complications or death", "Alive and well"],
    pdf_anemia,
)
# Dead or Complications vs Transfusion
get_fischer_df(
    patients,
    df_pvalues_transfu,
    column_transf[0],
    "DEAD_OR_COMPL",
    ["Transfusion", "No Transfusion"],
    ["Complications or death", "Alive and well"],
    pdf_transfu,
)

for compl in columns_compl:
    # One complication vs Anemia
    get_fischer_df(
        patients, df_pvalues_anemia, "ANEMIE", compl, ["Anaemic", "Not Anaemic"], [compl, f"No {compl}"], pdf_anemia
    )
    # One complication vs Transfusion
    get_fischer_df(
        patients,
        df_pvalues_transfu,
        column_transf[0],
        compl,
        ["Transfusion", "No Transfusion"],
        [compl, f"No {compl}"],
        pdf_transfu,
    )

df_pvalues_anemia = df_pvalues_anemia.applymap(format_x).astype(str)
df_pvalues_transfu = df_pvalues_transfu.applymap(format_x).astype(str)

pdf_anemia.add_table(TableWriter(data=df_pvalues_anemia))
pdf_transfu.add_table(TableWriter(data=df_pvalues_transfu))

"""
Logistic Regressions Male / Female
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

"""
Logistic Regressions Fragilite
"""

# Alive vs Anemia
make_logistic(patients, "DEAD", pdf_anemia, "FRAGILE")

# Complications vs Anemia
make_logistic(patients, "COMPL", pdf_anemia, "FRAGILE")

# Dead or Complications vs Anemia
make_logistic(patients, "DEAD_OR_COMPL", pdf_anemia, "FRAGILE")

# One complication vs Anemia
for compl in columns_compl:
    make_logistic(patients, compl, pdf_anemia, "FRAGILE")
