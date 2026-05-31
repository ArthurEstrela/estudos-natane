# =========================================
#  CARREGAR PACOTES
# =========================================
library(readxl)
library(ggplot2)
library(factoextra)
library(dplyr)
library(reshape2)

# =========================================
#  CARREGAR DADOS
# =========================================
dados <- read_excel("cheese.xls")

# Corrigir nomes
names(dados) <- make.names(names(dados))

# =========================================
#  SELECIONAR COLUNAS SENSORIAIS
# =========================================
dados_num <- dados %>%
  select(matches("N.|E.|M."))

# =========================================
#  CONVERTER PARA NUMÉRICO (CORRETO)
# =========================================
dados_num <- dados_num %>%
  mutate(across(everything(), ~as.numeric(as.character(.))))

# =========================================
#  REMOVER COLUNAS MUITO RUINS
# =========================================
dados_num <- dados_num[, colSums(is.na(dados_num)) < (0.5 * nrow(dados_num))]

# =========================================
#  REMOVER NA
# =========================================
dados_num <- na.omit(dados_num)

# =========================================
#  GARANTIA
# =========================================
if(nrow(dados_num) < 10){
  stop("Poucos dados após limpeza")
}

# =========================================
#  PADRONIZAÇÃO
# =========================================
dados_scaled <- scale(dados_num)

# =========================================
#  MÉTODO DO COTOVELO
# =========================================
fviz_nbclust(dados_scaled, kmeans, method = "wss") +
  ggtitle("Método do Cotovelo")

# =========================================
#  K-MEANS
# =========================================
set.seed(123)
k <- 3

kmeans_result <- kmeans(dados_scaled, centers = k, nstart = 25)

# =========================================
# DATASET FINAL (SEM ERRO)
# =========================================
dados_final <- as.data.frame(dados_num)
dados_final$Cluster <- as.factor(kmeans_result$cluster)

# =========================================
# TABELA
# =========================================
cluster_table <- dados_final %>%
  group_by(Cluster) %>%
  summarise(Quantidade = n())

print(cluster_table)

# =========================================
#  GRÁFICO CLUSTERS
# =========================================
fviz_cluster(kmeans_result,
             data = dados_scaled,
             geom = "point",
             ellipse.type = "convex",
             ggtheme = theme_minimal())

# =========================================
#  PCA
# =========================================
pca_result <- prcomp(dados_scaled)

fviz_pca_ind(pca_result,
             geom = "point",
             col.ind = dados_final$Cluster,
             addEllipses = TRUE)

# =========================================
# VARIÁVEIS IMPORTANTES
# =========================================
fviz_pca_var(pca_result,
             col.var = "contrib",
             repel = TRUE)

# =========================================
#  PERFIL DOS CLUSTERS
# =========================================
cluster_summary <- dados_final %>%
  group_by(Cluster) %>%
  summarise(across(everything(), mean))

print(cluster_summary)

# =========================================
#  HEATMAP
# =========================================
heat_data <- melt(cluster_summary, id.vars = "Cluster")

ggplot(heat_data, aes(x = variable, y = Cluster, fill = value)) +
  geom_tile() +
  theme_minimal() +
  labs(title = "Perfil dos Clusters",
       x = "Variáveis",
       y = "Cluster") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))