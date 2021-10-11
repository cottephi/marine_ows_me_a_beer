# Marine ows me a beer

## To do next

J'ai fait les analyses de fischer (**tables_hb.pdf** et **tables_transfu.pdf** produits par analyse_fischer.py), rien de plus à faire là dessus.

Les regressions log sont dans les mêmes fichiers, mais n'ont donné aucun resultat probant

J'ai codé l'analyse de Cox, qui fonctionne, mais qui ne répond pas à la problématique pour le moment. En effet, elle utilise tous les patients d'un coup,
alors que Marine aimerait les séparer en deux groupes : anémiques et pas anémiques.
La colonne "ANEMIE" du dataframe **patients** dans analyse_cox.py permet de faire ça.

De plus, il faudrait voir si il existe une méthode d'analyse plus pertinente pour nos données que l'analyse de Cox, plus adaptée aux 
variables explicatives continues alors que les notres sont discrètes.

Enfin, sur l'analyse de Cox, il faudra revoir la méthode de Censure. En effet, j'ai censuré à 30j tous les patients dont on n'a aucune info
sur la date de décès, alors que parmis ces patients il y en a que l'on sait morts (on ne sait juste pas quand ils sont morts), alors que d'autres sont
vivant, et d'autre encore pour lesquels on ne sait rien.

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

There are **226** patients with no fragility information 

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