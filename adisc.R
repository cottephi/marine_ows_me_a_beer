# Chargement des données
library(readxl)
POSE <- read_excel("Desktop/projets/thesemarine/POSE.xlsx")

# Nettoyage
colnames(POSE) <- POSE[1,]
data <- subset(POSE, subset=c(FALSE, rep(TRUE, 2252)))
rownames(data) <- data$`Study Subject ID`
data$`Study Subject ID` <- NULL

data$POIDS <- as.numeric(data$POIDS)
data$AGE <- as.numeric(data$AGE)
data$TAILLE <- as.numeric(data$TAILLE)
data$HEMOGLOBINE_PREOP <- as.numeric(data$HEMOGLOBINE_PREOP)
data$ALIVE.J30 <- as.factor(data$ALIVE.J30)
data$ALIVE.J30[data$ALIVE.J30=="NA"] <- NA

data <- subset(data, subset = !is.na(data$HEMOGLOBINE_PREOP))
data <- subset(data, subset = !is.na(data$ALIVE.J30))

# On sélectionne les variables qualitatives a priori intéressantes
data <- subset(data, select=c(14:28,30:36,38:43,45,48:50,52:56,64,65,68,72:78,80,81,87,81,83,86:89,93,96,98,99,118:135,137))

# On supprime celles qui n'ont qu'un seul niveau ou trop spécifiques
apply(data, 2, unique)
data$COMP.ACR <- NULL
data$NSQ_Cardiac.arrest <- NULL
data$INHOSPITAL.death <- NULL
data$DISCHARGE.before.J30 <- NULL
data$ACS.NSQ_comp <- NULL
data$Surgical.procedure <- NULL

# Analyse de Correspondances Multiples
library(DescTools)
library(FactoMineR)
library(factoextra)

data_acm <- data[,1:72]
res.acm <- MCA(data_acm, graph=F, ncp=2) # deux axes => deux critères composites

## On ajoute à chaque individu ses coordonnées sur le plan factoriel

data$x1 <- res.acm$ind$coord[,1]
data$x2 <- res.acm$ind$coord[,2]

## Il faut maintenant mener une analyse discriminante sur ces coordonnées par rapport à la variable anémie
