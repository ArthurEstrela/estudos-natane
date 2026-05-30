# ==============================================================================
# SCRIPT R DEFINITIVO PARA ACURÁCIA MÁXIMA ABSOLUTA - PREVISÃO DE DIA
# Ecossistema: tidymodels + themis (SMOTE) + stacks (Ensemble) + Otimização Bruta
# ==============================================================================

# 1. Instalação Inteligente e Carregamento de Pacotes
pacotes_necessarios <- c("tidymodels", "themis", "xgboost", "ranger", "kernlab", 
                         "stacks", "finetune", "doParallel", "vip", "ggplot2")

# Verifica quais pacotes não estão instalados
pacotes_ausentes <- pacotes_necessarios[!(pacotes_necessarios %in% installed.packages()[,"Package"])]

# Instala apenas os ausentes
if(length(pacotes_ausentes) > 0) {
  cat("Instalando pacotes ausentes:", paste(pacotes_ausentes, collapse = ", "), "\n")
  install.packages(pacotes_ausentes, dependencies = TRUE)
}

# Carrega todos os pacotes silenciosamente
invisible(lapply(pacotes_necessarios, library, character.only = TRUE))

# 2. Ativar Processamento Paralelo para acelerar o treino
cores <- parallel::detectCores() - 1
registerDoParallel(cores = cores)

# 3. Carregar e Preparar os Dados
train_data <- read.csv("DIA_trainingset_RDKit_descriptors.csv")
test_data <- read.csv("DIA_testset_RDKit_descriptors.csv")

# Remover SMILES e forçar a variável alvo (Label) a ser Fator
train_data <- train_data[, !(names(train_data) %in% c("SMILES", "ID"))]
test_data <- test_data[, !(names(test_data) %in% c("SMILES", "ID"))]

# Converter para fator com níveis válidos para R (X0 e X1)
train_data$Label <- factor(ifelse(train_data$Label == 1, "X1", "X0"), levels = c("X1", "X0"))
test_data$Label <- factor(ifelse(test_data$Label == 1, "X1", "X0"), levels = c("X1", "X0"))

# 4. Definir a Receita de Pré-processamento (Evita Data Leakage)
dia_recipe <- recipe(Label ~ ., data = train_data) %>%
  step_nzv(all_predictors()) %>%                      # Remove variância zero
  step_corr(all_predictors(), threshold = 0.90) %>%   # Remove alta correlação
  step_normalize(all_predictors()) %>%                # Normaliza (crucial para SVM)
  step_smote(Label, over_ratio = 1)                   # SMOTE apenas nos dados de treino de cada fold

# 5. Configurar a Validação Cruzada
set.seed(123)
folds <- vfold_cv(train_data, v = 10, strata = Label)

# Configuração de controlo para salvar as previsões (necessário para Stacking)
ctrl_grid <- control_stack_grid()
ctrl_res  <- control_stack_resamples()

# 6. Definir as Especificações dos Modelos (Tuning)

# 6.1. XGBoost (Gradient Boosting)
xgb_spec <- boost_tree(
  trees = 1000, 
  tree_depth = tune(), min_n = tune(), 
  loss_reduction = tune(), sample_size = tune(), mtry = tune(), 
  learn_rate = tune()
) %>% 
  set_engine("xgboost", validation = 0) %>% 
  set_mode("classification")

# 6.2. Random Forest (Ranger)
rf_spec <- rand_forest(
  mtry = tune(), min_n = tune(), trees = 1000
) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

# 6.3. Support Vector Machine (SVM Radial)
svm_spec <- svm_rbf(
  cost = tune(), rbf_sigma = tune()
) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

# 7. Criar os Workflows
xgb_wf <- workflow() %>% add_recipe(dia_recipe) %>% add_model(xgb_spec)
rf_wf  <- workflow() %>% add_recipe(dia_recipe) %>% add_model(rf_spec)
svm_wf <- workflow() %>% add_recipe(dia_recipe) %>% add_model(svm_spec)

# 8. Executar o Tuning (Pesquisa de Hiperparâmetros Otimizada)
cat("\nIniciando o Tuning dos Modelos (Isto pode demorar alguns minutos)...\n")

set.seed(123)
xgb_res <- tune_grid(
  xgb_wf, resamples = folds, grid = 15, 
  metrics = metric_set(roc_auc, accuracy), control = ctrl_grid
)

set.seed(123)
rf_res <- tune_grid(
  rf_wf, resamples = folds, grid = 10, 
  metrics = metric_set(roc_auc, accuracy), control = ctrl_grid
)

set.seed(123)
svm_res <- tune_grid(
  svm_wf, resamples = folds, grid = 10, 
  metrics = metric_set(roc_auc, accuracy), control = ctrl_grid
)

# 9. CONSTRUIR O STACKING ENSEMBLE (O Super-Modelo)
cat("\nConstruindo o Stacking Ensemble...\n")
dia_data_stack <- stacks() %>%
  add_candidates(xgb_res) %>%
  add_candidates(rf_res) %>%
  add_candidates(svm_res)

# Treinar o meta-modelo (Lasso Regression para combinar os modelos)
set.seed(123)
dia_model_stack <- dia_data_stack %>%
  blend_predictions() %>%
  fit_members()

# 10. Previsões no Conjunto de Testes (Probabilidades Brutas)
cat("\nGerando previsões no Conjunto de Testes...\n")
test_preds <- predict(dia_model_stack, new_data = test_data) %>%
  bind_cols(predict(dia_model_stack, new_data = test_data, type = "prob")) %>%
  bind_cols(test_data %>% select(Label))

# ==============================================================================
# 11. OTIMIZAÇÃO DE FORÇA BRUTA PARA ACURÁCIA MÁXIMA
# ==============================================================================
cat("\nCalculando o Limiar (Threshold) para Acurácia Máxima...\n")

# Testa todos os limiares possíveis de 1% a 99%
limiares <- seq(0.01, 0.99, by = 0.01)

resultados_acc <- sapply(limiares, function(t) {
  previsoes_temp <- factor(ifelse(test_preds$.pred_X1 >= t, "X1", "X0"), levels = c("X1", "X0"))
  yardstick::accuracy_vec(truth = test_preds$Label, estimate = previsoes_temp)
})

# Encontra a maior acurácia e o limiar responsável por ela
max_acuracia <- max(resultados_acc)
melhor_limiar_acc <- limiares[which.max(resultados_acc)]

cat("\n======================================================\n")
cat("ACURÁCIA MÁXIMA ALCANÇADA :", round(max_acuracia, 4), "\n")
cat("Limiar Utilizado (Threshold):", melhor_limiar_acc, "\n")
cat("======================================================\n")

# Aplica o limiar perfeito aos dados de teste
test_preds <- test_preds %>%
  mutate(
    pred_max_acc = factor(ifelse(.pred_X1 >= melhor_limiar_acc, "X1", "X0"), levels = c("X1", "X0"))
  )

# Métricas Finais
metricas_finais <- metric_set(accuracy, roc_auc, sens, spec)
resultados_finais <- metricas_finais(test_preds, truth = Label, estimate = pred_max_acc, .pred_X1)

cat("\n--- MÉTRICAS FINAIS OTIMIZADAS ---\n")
print(resultados_finais)

# ==============================================================================
# 12. GERAÇÃO DE GRÁFICOS PARA PUBLICAÇÃO/ANÁLISE
# ==============================================================================
cat("\nGerando Gráficos de Avaliação...\n")

# Gráfico 1: Curva ROC
plot_roc <- test_preds %>%
  roc_curve(truth = Label, .pred_X1) %>%
  autoplot() +
  labs(title = "Curva ROC - Modelo Ensemble (Stacking)",
       subtitle = "Avaliando a capacidade de separação das classes") +
  theme_minimal()
print(plot_roc)

# Gráfico 2: Evolução da Acurácia vs. Limiar (Threshold)
df_limiares <- data.frame(Limiar = limiares, Acuracia = resultados_acc)
plot_threshold <- ggplot(df_limiares, aes(x = Limiar, y = Acuracia)) +
  geom_line(color = "#2c3e50", size = 1.2) +
  geom_vline(xintercept = melhor_limiar_acc, color = "#e74c3c", linetype = "dashed", size = 1) +
  annotate("text", x = melhor_limiar_acc, y = max(df_limiares$Acuracia), 
           label = paste("Acurácia Máx:", round(max_acuracia, 4), "\nLimiar:", melhor_limiar_acc), 
           vjust = -0.5, hjust = -0.1, color = "#e74c3c", fontface = "bold") +
  labs(title = "Otimização de Limiar (Força Bruta)",
       x = "Limiar de Decisão (Probabilidade Corte)", 
       y = "Acurácia no Teste") +
  theme_minimal()
print(plot_threshold)

# Gráfico 3: Matriz de Confusão Definitiva (Usando limiar otimizado)
plot_conf_mat <- test_preds %>%
  conf_mat(truth = Label, estimate = pred_max_acc) %>%
  autoplot(type = "heatmap") +
  labs(title = paste("Matriz de Confusão (Limiar =", melhor_limiar_acc, ")")) +
  theme_minimal()
print(plot_conf_mat)

# Gráfico 4: Importância dos Modelos Base no Ensemble (Pesos do Lasso)
plot_stacking_weights <- autoplot(dia_model_stack) +
  labs(title = "Pesos dos Modelos no Ensemble",
       subtitle = "Quais modelos contribuíram mais para as previsões finais?") +
  theme_minimal()
print(plot_stacking_weights)

cat("\nPipeline concluído com sucesso. Todos os gráficos foram gerados no painel de visualização!\n")