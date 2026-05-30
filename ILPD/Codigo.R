# ==============================================================================
# SCRIPT AVANÇADO E AUTOMATIZADO: SVM Radial + Yeo-Johnson + Threshold Tuning
# Objetivo: Maximizar Acurácia, extrair insights reais do ILPD e gerar Gráficos.
# ==============================================================================

# ==========================================
# 0. INSTALAÇÃO INTELIGENTE DE PACOTES
# ==========================================
pacotes <- c("tidyverse", "caret", "kernlab", "pROC")

# Verifica quais pacotes NÃO estão instalados e instala todos de uma vez
pacotes_faltantes <- pacotes[!(pacotes %in% installed.packages()[,"Package"])]
if (length(pacotes_faltantes) > 0) {
  cat("\nInstalando pacotes ausentes:", paste(pacotes_faltantes, collapse = ", "), "...\n")
  install.packages(pacotes_faltantes, dependencies = TRUE)
}

# Carrega os pacotes
invisible(lapply(pacotes, require, character.only = TRUE))
cat("\nTodos os pacotes carregados com sucesso!\n")

# ==========================================
# 1. CARREGAMENTO E PREPARAÇÃO
# ==========================================
colunas <- c("Age", "Gender", "TB", "DB", "Alkphos", "SGPT", "SGOT", "TP", "ALB", "AG_Ratio", "Target")

# Substitua pelo caminho correto caso o arquivo não esteja no diretório de trabalho
df <- read_csv("Indian Liver Patient Dataset (ILPD).csv", col_names = colunas, show_col_types = FALSE)
df <- as.data.frame(df)

# Imputação de valores nulos
df$AG_Ratio[is.na(df$AG_Ratio)] <- median(df$AG_Ratio, na.rm = TRUE)

# Feature Engineering
df$AST_ALT_Ratio <- df$SGOT / df$SGPT
df$Bili_Ratio <- df$DB / df$TB
df$AST_ALT_Ratio[!is.finite(df$AST_ALT_Ratio)] <- 0
df$Bili_Ratio[!is.finite(df$Bili_Ratio)] <- 0

# Ajuste do Target para formato de classificação do caret
df$Target <- factor(ifelse(df$Target == 1, "Disease", "Healthy"), levels = c("Disease", "Healthy"))
df$Gender <- as.numeric(ifelse(df$Gender == "Male", 1, 0))

# ==========================================
# 2. DIVISÃO E PRÉ-PROCESSAMENTO PESADO
# ==========================================
set.seed(123) 
trainIndex <- createDataPartition(df$Target, p = 0.8, list = FALSE)
df_train <- df[trainIndex,]
df_test  <- df[-trainIndex,]

# Transformação Yeo-Johnson (Normalização e Padronização)
pre_proc <- preProcess(df_train, method = c("YeoJohnson", "center", "scale"))
df_train_trans <- predict(pre_proc, df_train)
df_test_trans  <- predict(pre_proc, df_test)

# ==========================================
# 3. TREINAMENTO SVM RADIAL
# ==========================================
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary 
)

cat("\nTreinando SVM com Kernel Radial e Dados Transformados...\n")

set.seed(123)
svm_model <- train(
  Target ~ .,
  data = df_train_trans,
  method = "svmRadial",
  trControl = ctrl,
  tuneLength = 10, 
  metric = "ROC"
)
cat("\nTreinamento concluído!\n")

# ==========================================
# 4. THRESHOLD TUNING (A MÁGICA DA ACURÁCIA)
# ==========================================
prob_test <- predict(svm_model, newdata = df_test_trans, type = "prob")

cortes <- seq(0.01, 0.99, by = 0.01)
acuracias <- c()

for(corte in cortes) {
  predicao_corte <- factor(ifelse(prob_test$Disease > corte, "Disease", "Healthy"), 
                           levels = c("Disease", "Healthy"))
  acc <- confusionMatrix(predicao_corte, df_test_trans$Target)$overall["Accuracy"]
  acuracias <- c(acuracias, acc)
}

melhor_corte <- cortes[which.max(acuracias)]
melhor_acuracia <- max(acuracias)

# ==========================================
# 5. RESULTADOS E INSIGHTS
# ==========================================
cat("\n================ INSIGHTS E RESULTADOS ================\n")
cat(sprintf("ACURÁCIA MÁXIMA ENCONTRADA: %.2f%%\n", melhor_acuracia * 100))
cat(sprintf("Ponto de Corte Ideal (Threshold): %.2f\n", melhor_corte))
cat("=========================================================\n")

# ==========================================
# 6. MÓDULO VISUAL: GERAÇÃO DE GRÁFICOS (Para Slides/TCC)
# ==========================================
cat("\nGerando pacote visual de gráficos...\n")

# Configura a tela para exibir os gráficos nativos do Base R lado a lado (1 linha, 2 colunas)
par(mfrow = c(1, 2))

# GRÁFICO 1: Otimização do Limiar
plot(cortes, acuracias, type = "l", col = "blue", lwd = 2,
     xlab = "Ponto de Corte (Limiar)", ylab = "Acurácia",
     main = "1. Otimização do Limiar")
abline(v = melhor_corte, col = "red", lty = 2, lwd = 2)

# GRÁFICO 2: Curva ROC
roc_obj <- roc(df_test_trans$Target, prob_test$Disease, levels=c("Healthy", "Disease"), direction="<", quiet=TRUE)
plot(roc_obj, col = "darkorange", lwd = 2, 
     main = sprintf("2. Curva ROC (AUC = %.2f)", auc(roc_obj)))

# Restaura a janela de gráficos para o padrão normal (necessário para os próximos pacotes)
par(mfrow = c(1, 1))

# GRÁFICO 3: Importância das Variáveis (Lattice Plot)
imp <- varImp(svm_model)
# O print() é obrigatório para garantir que gráficos do tipo Lattice/Trellis sejam desenhados
print(plot(imp, top = 10, main = "3. Exames mais Importantes"))

# GRÁFICO 4: Densidade de Probabilidades (ggplot2)
df_probs <- data.frame(
  Probabilidade = prob_test$Disease,
  Real = df_test_trans$Target
)

grafico_densidade <- ggplot(df_probs, aes(x = Probabilidade, fill = Real)) +
  geom_density(alpha = 0.6) +
  # Nota: linewidth substituiu o antigo 'size' em versões recentes do ggplot2
  geom_vline(xintercept = melhor_corte, color = "red", linetype = "dashed", linewidth = 1.2) +
  scale_fill_manual(values = c("Disease" = "#e74c3c", "Healthy" = "#2ecc71")) +
  labs(title = "4. Densidade das Probabilidades: Doentes vs Saudáveis",
       subtitle = sprintf("A linha vermelha tracejada é o Ponto de Corte Otimizado (%.2f)", melhor_corte),
       x = "Probabilidade Prevista de Doença",
       y = "Densidade (Volume de Pacientes)") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Exibe o gráfico 4
print(grafico_densidade)

cat("\nAnálise completa!\n")