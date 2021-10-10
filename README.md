# Marine ows me a beer

## Data

There are 2252 patients

There are 1445 patients with available Hb data

There are 1239 patients that are still alive from ALIVE.J30

There are 7 patients that are dead from ALIVE.J30

We found an addition of 32 dead patients from their date information

We found an addition of 0 living patients from their date information

There are 167 patients with no death information at all. Ignoring them.

Total : 1239 alive and 39 dead

There are 544 anaemic patients

There are 215 living patients with at least one POST-OP complication

There are 220 patients dead or with at least one POST-OP complication

## Exact Fischer Test : death/complications vs anemia or transfusion

See **tables_hb.pdf** and **tables_transfu** for results.


## Logistic regression

### Death/complications vs Hb

See **tables_hb.pdf** for results.

Vertical lines are the limit under which a male (13 g/dL) or a female (12 g/dL) patient is considered anaemic.

### Cox regression

Concernant la regression de Cox, l'idée est d'avoir un modèle capable de prédire dans combien de temps va survenir un
événement à partir de variables explicatives. 

Pour entraîner un tel modèle, il faut des variables explicatives (dans notre cas, anémie et fragilités par exemple) et
la date à laquelle est survenue un événement (dans notre cas, le nombre de jours entre la mort et l'opération,
ou entre la mort et une complication).

Le problème est que l'on a cette information pour 38 patients seulement (nombre de jours entre la date de décès et la
date induction), puisque pour les autres on a qu'une information binaire (mort avant 30j ou non). 38 patients c'est 
trop peu pour un fit je pense, d'autant qu'on veut plusieurs critères explicatifs.

Concernant les complications, je crois bien qu'on a aucune 
information temporelle.