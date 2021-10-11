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
