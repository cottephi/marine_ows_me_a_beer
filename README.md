# Marine ows me a beer

## Exact Fischer Test

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

|             | Alive     | Dead |
|-------------|-----------|------|
| Anaemic     | 517       | 27   |
| Not Anaemic | 722       | 12   |
| one/two -tail P-value   | 0.32 | 8.2x10<sup>-4</sup> |

|             | Complications     | Alive and well |
|-------------|-----------|------|
| Anaemic     | 106       | 438   |
| Not Anaemic | 75        | 659   |
| one/two -tail P-value   | 2.1 | 3.3x10<sup>-6</sup> |

|             | Dead or complications     | Alive and well |
|-------------|-----------|------|
| Anaemic     | 133       | 411   |
| Not Anaemic | 87       | 647   |
| one/two -tail P-value | 2.4 | 5.8x10<sup>-9</sup> |


## Logistic regression

### Death vs Hb

![](death_vs_hb.png)

Vertical lines are the limit under which a male (13 g/dL) or a female (12 g/dL) patient is considered anaemic.
