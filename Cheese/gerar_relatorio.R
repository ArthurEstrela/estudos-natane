# ==========================================================
#  GERAÇÃO DE GRÁFICOS E MÉTRICAS PARA O RELATÓRIO - BASE CHEESE
#  (Espelha ScriptExtra.R, mas salva os gráficos em PNG)
# ==========================================================
options(warn = -1)

pacotes <- c("readxl", "ggplot2", "factoextra", "dplyr", "reshape2")
faltantes <- pacotes[!(pacotes %in% installed.packages()[, "Package"])]
if (length(faltantes) > 0) {
  install.packages(faltantes, repos = "https://cloud.r-project.org")
}
invisible(lapply(pacotes, library, character.only = TRUE))

# --- diretório de saída ---
dir_out <- "fig"
if (!dir.exists(dir_out)) dir.create(dir_out)

# --- carregar dados ---
dados <- read_excel("cheese.xls")
names(dados) <- make.names(names(dados))

cat("=== DIMENSAO ORIGINAL ===\n")
cat("Linhas:", nrow(dados), "| Colunas:", ncol(dados), "\n")
cat("=== NOMES DAS COLUNAS ===\n")
print(names(dados))

# --- seleção sensorial (igual ao ScriptExtra.R) ---
dados_num <- dados %>% select(matches("N.|E.|M."))
dados_num <- dados_num %>% mutate(across(everything(), ~ as.numeric(as.character(.))))
dados_num <- dados_num[, colSums(is.na(dados_num)) < (0.5 * nrow(dados_num))]
dados_num <- na.omit(dados_num)

cat("\n=== APOS LIMPEZA ===\n")
cat("Linhas:", nrow(dados_num), "| Variaveis sensoriais:", ncol(dados_num), "\n")
cat("Variaveis usadas:\n"); print(names(dados_num))

# --- padronização ---
dados_scaled <- scale(dados_num)

# --- 1. Método do Cotovelo ---
g1 <- fviz_nbclust(dados_scaled, kmeans, method = "wss") +
  ggtitle("1. Método do Cotovelo (Elbow)") +
  theme_minimal(base_size = 13)
ggsave(file.path(dir_out, "01_cotovelo.png"), g1, width = 7, height = 5, dpi = 150)

# --- K-Means ---
set.seed(123)
k <- 3
km <- kmeans(dados_scaled, centers = k, nstart = 25)

cat("\n=== RESULTADO K-MEANS (k=3) ===\n")
cat("Tamanho dos clusters:", paste(km$size, collapse = ", "), "\n")
cat("Soma de quadrados intra (within):", round(km$tot.withinss, 2), "\n")
cat("Soma de quadrados entre (between):", round(km$betweenss, 2), "\n")
cat("Razao between/total:", round(km$betweenss / km$totss * 100, 1), "%\n")

dados_final <- as.data.frame(dados_num)
dados_final$Cluster <- as.factor(km$cluster)

# --- 2. Clusters (espaço reduzido) ---
g2 <- fviz_cluster(km, data = dados_scaled, geom = "point",
                   ellipse.type = "convex", ggtheme = theme_minimal(base_size = 13)) +
  ggtitle("2. Visualização dos Clusters (K-Means, k=3)")
ggsave(file.path(dir_out, "02_clusters.png"), g2, width = 7, height = 5, dpi = 150)

# --- PCA ---
pca <- prcomp(dados_scaled)
var_exp <- round(summary(pca)$importance[2, 1:2] * 100, 1)
cat("\n=== PCA ===\n")
cat("Variancia explicada PC1:", var_exp[1], "% | PC2:", var_exp[2], "%\n")
cat("Acumulada (PC1+PC2):", sum(var_exp), "%\n")

# --- 3. Contribuição das variáveis (PCA) ---
g3 <- fviz_pca_var(pca, col.var = "contrib", repel = TRUE,
                   gradient.cols = c("#cccccc", "#377eb8", "#e41a1c")) +
  ggtitle("3. Contribuição das Variáveis (PCA)") +
  theme_minimal(base_size = 13)
ggsave(file.path(dir_out, "03_pca_var.png"), g3, width = 7, height = 5, dpi = 150)

# --- perfil dos clusters + heatmap ---
cluster_summary <- dados_final %>%
  group_by(Cluster) %>%
  summarise(across(everything(), mean))

cat("\n=== PERFIL MEDIO DOS CLUSTERS ===\n")
print(as.data.frame(round(cluster_summary[-1], 2)))

heat <- melt(cluster_summary, id.vars = "Cluster")
g4 <- ggplot(heat, aes(x = variable, y = Cluster, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "#377eb8", mid = "#f7f7f7", high = "#e41a1c", midpoint = 0) +
  theme_minimal(base_size = 13) +
  labs(title = "4. Perfil dos Clusters (Heatmap)", x = "Variáveis Sensoriais", y = "Cluster") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(dir_out, "04_heatmap.png"), g4, width = 8, height = 5, dpi = 150)

cat("\n=== TAMANHO DOS CLUSTERS (TABELA) ===\n")
print(as.data.frame(table(Cluster = dados_final$Cluster)))

cat("\nGráficos salvos em Cheese/fig/\n")
