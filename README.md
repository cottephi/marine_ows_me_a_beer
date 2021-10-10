# Marine ows me a beer

## Data

There are **2252** patients 

There are **1984** patients that are still alive from ALIVE.J30.
There are **9** patients that are dead from ALIVE.J30.
There are **259** patients with no ALIVE.J30 information 

We found an addition of **33** dead patients from their date information.
We found an addition of **0** living patients from their date information 

Total : **1984** alive, **42** dead, **226** without information 

There are **41** patients with death date information
There are **2211** patients with no death date information 

There are **544** anaemic patients.
There are **734** not anaemic patients.
There are **974** patients without Hb informations 

There are **248** patients with at least one POST-OP complication.
There are **226** patients with no useable information about POST-OP complication 

There are **254** patients dead or with at least one POST-OP complication.
There are **1772** patients alive and well 

There are **226** patients with fragility information 

There are **226** patients with no age information 

There are **226** patients with no General Anaesthesis information 

There are **226** patients with no chir. severity information 

There are **226** patients with no information about operation programmation 


## Exact Fischer Test : death/complications vs anemia or transfusion

See **tables_hb.pdf** and **tables_transfu** for results.


## Logistic regression

### Death/complications vs Hb

See **tables_hb.pdf** for results.

Vertical lines are the limit under which a male (13 g/dL) or a female (12 g/dL) patient is considered anaemic.

### Cox regression

Il n'y a que 39 patients pour lesquels on connait le nombre de jour entre l'opération et le décès.

Variables explicatives :
 * Age
 * Sévérité de la chirurgie
 * Anésthésie : générale ou non
 * type de chirurgie : urgente ou non
 * fragilités : >4 ou non
 * IRA : présent ou non
 * Score ASA (1-5)