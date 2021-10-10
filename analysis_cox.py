from matplotlib import pyplot as plt
from pdffactory import PdfFactory
from tablewriter import TableWriter

from functions import (
    get_data,
    set_time_of_death,
    set_anemia,
    set_compl,
    set_alive,
    set_dead_or_compl,
    set_binned_age,
    set_binned_sexe,
    set_ag,
    set_fragility,
    set_severity,
    set_prog,
    set_censure,
    set_morbidite,
    fill_cox,
    drop_for_cox,
    fit_cox
)

patients = get_data()
set_alive(patients)
set_time_of_death(patients)
set_anemia(patients)
set_compl(patients)
set_dead_or_compl(patients)
set_fragility(patients)
set_binned_age(patients)
set_binned_sexe(patients)
set_ag(patients)
set_severity(patients)
set_prog(patients)
set_morbidite(patients)
set_censure(patients)
drop_for_cox(patients)

df_cox = fill_cox(patients)
df_cox.to_csv("cox_filled.csv")

patients_date_only = patients[~patients["DEAD_DAYS"].isna()]

df_cox_date_only = fill_cox(patients_date_only)
df_cox_date_only.to_csv("cox_date_only_filled.csv")

model = fit_cox(patients)

model.print_summary()

pdf = PdfFactory("cox.pdf")

model.plot()
pdf.add_figure(plt.gcf())
pdf.add_table(TableWriter(data=model.summary))
