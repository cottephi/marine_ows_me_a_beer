from functions import (
    get_data,
    set_time_of_death,
    set_anemia,
    set_compl,
    set_alive,
    set_dead_or_compl,
    set_binned_age,
    set_ag,
    set_fragility,
    set_severity,
    set_prog,
    init_cox
)

patients = get_data()
set_alive(patients)
set_time_of_death(patients)
set_anemia(patients)
set_compl(patients)
set_dead_or_compl(patients)
set_fragility(patients)
set_binned_age(patients)
set_ag(patients)
set_severity(patients)
set_prog(patients)

df_cox = init_cox()
