# sem o SMOTE
if(!require(dplyr)) install.packages("dplyr")
if(!require(caret)) install.packages("caret")
if(!require(recipes)) install.packages("recipes")
if(!require(ranger)) install.packages("ranger")
if(!require(glmnet)) install.packages("glmnet") 
if(!require(kknn)) install.packages("kknn")     

library(dplyr)
library(caret)
library(recipes)
library(ranger)
library(glmnet)
library(kknn)

# 1. CARGA E LIMPEZA
dados <- read.csv("drug_consumption.data", header = FALSE, stringsAsFactors = FALSE)

colnames(dados) <- c("ID", "Age", "Gender", "Education", "Country", "Ethnicity",
                     "Nscore", "Escore", "Oscore", "Ascore", "Cscore", 
                     "Impulsive", "SS",
                     "Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", 
                     "Choc", "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine", 
                     "Legalh", "LSD", "Meth", "Mushrooms", "Nicotine", "Semer", "VSA")

dados <- dados %>% filter(Semer == "CL0") %>% select(-Semer, -ID)

# 2. CONFIGURAÇÃO
lista_drogas <- c("Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", 
                  "Choc", "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine", 
                  "Legalh", "LSD", "Meth", "Mushrooms", "Nicotine", "VSA")

# Tabela: Métricas de Performance (Sem dados do SMOTE)
tabela_metricas <- data.frame(
  Droga = character(), 
  Melhor_Modelo = character(),
  Acuracia = numeric(), 
  Sensibilidade = numeric(), 
  Especificidade = numeric(), 
  F1_Score = numeric(),        
  stringsAsFactors = FALSE
)

print("Iniciando Competição de Algoritmos (Ranger vs Glmnet vs KNN) - SEM SMOTE...")

# 3. LOOP
for (droga_atual in lista_drogas) {
  
  df_temp <- dados %>%
    select(Age:SS, all_of(droga_atual)) %>%
    rename(Classe = all_of(droga_atual))
  
  df_temp$Classe <- factor(ifelse(df_temp$Classe %in% c("CL0", "CL1"), "Nao_Usa", "Usa"),
                           levels = c("Nao_Usa", "Usa"))
  
  set.seed(123)
  index <- createDataPartition(df_temp$Classe, p = 0.75, list = FALSE)
  treino_raw <- df_temp[index, ] 
  teste      <- df_temp[-index, ]
  
  # RECEITA: Apenas a Normalização obrigatória para Glmnet e KNN (SMOTE Removido)
  rec <- recipe(Classe ~ ., data = treino_raw) %>%
    step_normalize(all_numeric_predictors())
  
  prep_rec <- prep(rec, training = treino_raw)
  treino_final <- bake(prep_rec, new_data = NULL)
  teste_final  <- bake(prep_rec, new_data = teste) 
  
  ctrl <- trainControl(method = "cv", number = 5) 
  
  # --- TREINANDO OS 3 MODELOS ---
  
  # 1. Ranger
  grid_ranger <- expand.grid(mtry = c(2, 4), splitrule = "gini", min.node.size = 1)
  mod_ranger <- train(Classe ~ ., data = treino_final, method = "ranger", 
                      trControl = ctrl, tuneGrid = grid_ranger, num.trees = 200, importance = 'none')
  
  # 2. Glmnet (Elastic Net)
  grid_glm <- expand.grid(alpha = c(0, 0.5, 1), lambda = seq(0.001, 0.1, length = 5))
  mod_glm <- suppressWarnings(train(Classe ~ ., data = treino_final, method = "glmnet", 
                                    trControl = ctrl, tuneGrid = grid_glm))
  
  # 3. KNN
  mod_knn <- train(Classe ~ ., data = treino_final, method = "kknn", 
                   trControl = ctrl, tuneLength = 5)
  
  # --- AVALIANDO OS 3 MODELOS ---
  
  avaliar_modelo <- function(modelo, nome) {
    pred <- predict(modelo, teste_final)
    cm <- confusionMatrix(pred, teste_final$Classe, mode = "everything", positive = "Usa")
    acc  <- cm$overall['Accuracy'] * 100
    sens <- cm$byClass['Sensitivity'] * 100  
    spec <- cm$byClass['Specificity'] * 100  
    f1   <- cm$byClass['F1'] * 100            
    if(is.na(f1)) f1 <- 0 
    return(data.frame(Modelo = nome, Acc = acc, Sens = sens, Spec = spec, F1 = f1))
  }
  
  res_ranger <- avaliar_modelo(mod_ranger, "Ranger")
  res_glm    <- avaliar_modelo(mod_glm, "Glmnet")
  res_knn    <- avaliar_modelo(mod_knn, "KNN")
  
  # Juntando resultados e pegando o vencedor pela maior Acurácia
  resultados <- rbind(res_ranger, res_glm, res_knn)
  vencedor <- resultados %>% arrange(desc(Acc), desc(F1)) %>% slice(1)
  
  tabela_metricas[nrow(tabela_metricas) + 1, ] <- list(
    droga_atual, vencedor$Modelo, round(vencedor$Acc, 2), 
    round(vencedor$Sens, 2), round(vencedor$Spec, 2), round(vencedor$F1, 2)
  )
  
  print(paste("->", droga_atual,  
              "| Campeão:", vencedor$Modelo, 
              "| Acc:", round(vencedor$Acc, 2), 
              "| F1:", round(vencedor$F1, 2)))
}

# 4. RESULTADOS PARA O ARTIGO
print("=== TABELA 1: PERFORMANCE SEM SMOTE (COM O MELHOR ALGORITMO POR DROGA) ===")
print(tabela_metricas)


# =====================================================================
# CÓDIGO ADICIONAL: GERAÇÃO DE TABELAS E GRÁFICOS (VERSÃO SEM SMOTE)
# =====================================================================

# Instalando pacotes necessários caso não existam
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(tidyr)) install.packages("tidyr")
if(!require(knitr)) install.packages("knitr")

library(ggplot2)
library(tidyr)
library(knitr)

# 1. GERANDO A TABELA FORMATADA

cat("\n\n=== TABELA FINAL DE MÉTRICAS (SEM SMOTE) ===\n")
print(kable(tabela_metricas, format = "markdown", align = "c", 
            caption = "Métricas de Performance por Droga e Melhor Modelo (Sem Balanceamento)"))

# 2. PREPARANDO OS DADOS PARA O GRÁFICO

tabela_longa <- tabela_metricas %>%
  select(Droga, Acuracia, Sensibilidade, Especificidade, F1_Score) %>%
  pivot_longer(
    cols = c(Acuracia, Sensibilidade, Especificidade, F1_Score),
    names_to = "Metrica",
    values_to = "Valor"
  )

# 3. CRIANDO O GRÁFICO DE BARRAS

grafico_desempenho_sem_smote <- ggplot(tabela_longa, aes(x = reorder(Droga, Valor), y = Valor, fill = Metrica)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10, face = "bold"),
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    legend.position = "top",
    panel.grid.major.x = element_blank()
  ) +
  labs(
    title = "Performance dos Modelos Droga (SEM SMOTE)",
    subtitle = "Comparação de Acurácia, Sensibilidade, Especificidade e F1-Score",
    x = "Tipo de Droga",
    y = "Valor da Métrica (%)",
    fill = "Métrica:"
  ) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 10)) +
  scale_fill_manual(values = c("Acuracia" = "#2E7D32",         # Verde escuro
                               "Especificidade" = "#1565C0",   # Azul escuro
                               "Sensibilidade" = "#C62828",    # Vermelho
                               "F1_Score" = "#F9A825"))        # Amarelo

# 4. EXIBINDO O GRÁFICO NA TELA
print(grafico_desempenho_sem_smote)


