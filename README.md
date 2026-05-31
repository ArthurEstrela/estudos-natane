# 🧠 Mineração de Dados — Otimização Preditiva e Análise Multibase

> **Controle impositivo sobre os dados.** Um estudo aplicado de *Feature Engineering*, mitigação de *Data Leakage* e *Threshold Tuning* para maximização de acurácia.

<p align="left">
  <img src="https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white" alt="R"/>
  <img src="https://img.shields.io/badge/Caret-FF6F00?style=for-the-badge&logo=r&logoColor=white" alt="Caret"/>
  <img src="https://img.shields.io/badge/Tidymodels-CB3837?style=for-the-badge&logo=r&logoColor=white" alt="Tidymodels"/>
  <img src="https://img.shields.io/badge/Random%20Forest-2E7D32?style=for-the-badge" alt="Random Forest"/>
  <img src="https://img.shields.io/badge/XGBoost-007ACC?style=for-the-badge" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/SVM%20Radial-8E44AD?style=for-the-badge" alt="SVM"/>
  <img src="https://img.shields.io/badge/SMOTE-themis-C62828?style=for-the-badge" alt="SMOTE"/>
</p>

<p align="left">
  <img src="https://img.shields.io/badge/Disciplina-Temática%20em%20Mineração%20de%20Dados-blue?style=flat-square" alt="Disciplina"/>
  <img src="https://img.shields.io/badge/Curso-Sistemas%20de%20Informação-informational?style=flat-square" alt="Curso"/>
  <img src="https://img.shields.io/badge/Instituição-IF%20Goiano%20·%20Campus%20Urutaí-success?style=flat-square" alt="IF Goiano"/>
</p>

---

## 📌 Visão Geral e Objetivo

Este repositório consolida o trabalho desenvolvido na disciplina **Temática em Mineração de Dados**, parte da grade do curso de **Sistemas de Informação** do **Instituto Federal Goiano — Campus Urutaí**.

A disciplina foi conduzida no formato de uma **competição algorítmica**, organizada em **5 rodadas semanais**, cada uma com uma base de dados distinta. O objetivo central era **maximizar a Acurácia** e **otimizar as métricas preditivas** de cada conjunto, que trazia características, vieses e desafios próprios. Não se tratava apenas de "rodar um modelo" — era preciso extrair o máximo de desempenho possível dos dados.

Além das 5 rodadas de competição, o repositório reúne uma **base de dados extra** que amplia o escopo do trabalho: um **estudo não-supervisionado** de segmentação (clusterização de queijos), evidenciando domínio tanto de aprendizado supervisionado quanto não-supervisionado.

Cada base recebeu um pipeline dedicado em **R**, com pré-processamento, validação cruzada, balanceamento e ajuste fino de hiperparâmetros e limiares de decisão.

### 👥 Equipe

| Integrante | |
|---|---|
| **Arthur Faria** | Desenvolvimento dos pipelines, engenharia de atributos e otimização de limiares |
| **Ricardo Issa** | Modelagem, validação e análise de métricas |
| **Sávio Issa** | Modelagem, validação e análise de métricas |

---

## 🎯 Nossa Filosofia Técnica (O Diferencial)

Nossa estratégia para vencer a competição **não foi aceitar os resultados padrão dos algoritmos**. Partimos do princípio de que o modelo é apenas a última etapa — e a menos importante — de um pipeline bem construído. Adotamos o que chamamos de **"controle impositivo sobre os dados"**: cada decisão de pré-processamento foi deliberada, justificada e mensurada.

Esse controle se manifestou em três pilares:

1. **Feature Engineering com injeção de domínio** — criamos variáveis que os dados brutos não entregavam de graça (scores comportamentais, razões clínicas, indicadores de risco), traduzindo conhecimento do problema em sinal preditivo.
2. **Combate rigoroso ao Data Leakage** — o balanceamento sintético (SMOTE) e a normalização foram isolados **dentro das dobras de treino** da Validação Cruzada (via `recipes`/`tidymodels`), garantindo que nenhuma informação do conjunto de teste vazasse para o treino e inflasse artificialmente as métricas.
3. **Threshold Tuning por força bruta** — em vez de aceitar o limiar default de `0.50`, realizamos varreduras exaustivas no ponto de corte da probabilidade para encontrar matematicamente o limiar que maximiza a métrica-alvo de cada problema.

> **Em resumo:** não deixamos o algoritmo decidir sozinho. Nós impusemos a estrutura, controlamos o vazamento e calibramos o ponto de decisão.

---

## 🔬 Os Estudos de Caso

> A competição foi disputada em **5 rodadas semanais** (Rodadas 1 a 5). A **Base Extra - 06** é uma análise não-supervisionada complementar.

### 🏁 Rodada 1 de Competição (16/03) — Base do Cartão de Crédito (*Credit Card Default*)

📂 [`CartaoCredito/`](CartaoCredito/) · Script: [`cartaoCreditoARS.R`](CartaoCredito/cartaoCreditoARS.R) · Relatório: [`cartaoCredito_Relatorio.pdf`](CartaoCredito/cartaoCredito_Relatorio.pdf)

**O Desafio.** Identificar clientes que entrarão em **inadimplência** (*default*) no próximo mês — um problema clássico de classe desbalanceada, onde os "bons pagadores" dominam a base e mascaram os calotes.

**A Abordagem.**
- **Feature Engineering comportamental:** criamos o **"Score de Calote"** (`Total_Meses_Atraso`), que conta em quantos dos 6 meses históricos o cliente esteve inadimplente, e a **Razão Pagamento/Dívida** (`Racio_Pagamento_Divida`), comparando tudo que foi pago contra tudo que foi cobrado no período.
- **Higienização estatística:** aplicamos filtro de **variância quase-nula** (`nearZeroVar`) e remoção de **multicolinearidade** com `findCorrelation` em `cutoff = 0.90`, eliminando redundância antes da modelagem.
- **Seleção de variáveis (RFE):** *Recursive Feature Elimination* com validação cruzada de 10 dobras para filtrar ruído e reter apenas os preditores que realmente carregam sinal.
- **Modelo:** **Random Forest** com balanceamento **SMOTE** aplicado via `trainControl(sampling = "smote")`.

**O Racional Técnico — O Ponto Chave.**
O coração desta solução foi o **rebaixamento do limiar de decisão de `0.50` para `0.38`**. Com o corte padrão, o modelo era conservador demais e deixava passar inadimplentes reais. Ao baixar o limiar, aumentamos a **Sensibilidade (Recall)** — capturando mais calotes verdadeiros — e ajustamos conscientemente o *trade-off* entre Sensibilidade, Especificidade e Acurácia, visualizado num gráfico de performance por limiar.

```r
# Limiar otimizado: capturar mais inadimplentes reais
limiar_otimizado <- 0.38
pred_classes_38 <- factor(ifelse(pred_probs$Sim > limiar_otimizado, "Sim", "Nao"),
                          levels = c("Nao", "Sim"))
confusionMatrix(pred_classes_38, teste$Default, positive = "Sim")
```

---

### 🟡 Rodada 2 de Competição (23/03) — Base Gallstone (Pedra na Vesícula)

📂 [`Predição_doenca/`](Predição_doenca/) · Script: [`Codigo.R`](Predição_doenca/Codigo.R) · Relatório: [`DoencaDaVesicula.pdf`](Predição_doenca/DoencaDaVesicula.pdf)

**O Desafio.** Diagnosticar a presença de **cálculos biliares** (pedra na vesícula) a partir de um conjunto rico de exames laboratoriais e medidas de **bioimpedância** corporal — uma base de alta dimensionalidade clínica, na qual nem toda variável carrega sinal.

**A Abordagem.**
- **Higienização semântica:** padronização dos nomes de colunas com `janitor::clean_names`, garantindo um pipeline `tidymodels` limpo e legível.
- **Seleção dirigida por domínio:** em vez de jogar tudo no modelo, **selecionamos as 15 variáveis clínicas mais decisivas** (Proteína C-Reativa, Vitamina D, AST, HDL, percentual de gordura corporal, água extracelular, hemoglobina, creatinina, fosfatase alcalina, gordura visceral, entre outras) — uma curadoria que reduz ruído e melhora a generalização.
- **Receita à prova de vazamento (`recipe`):** imputação por mediana (`step_impute_median`), normalização (`step_normalize`) e remoção de multicolinearidade (`step_corr` em `0.90`), tudo estimado apenas no treino.
- **Modelo:** **Random Forest (Ranger)** com `mtry` e `min_n` otimizados via `grid_regular` + **Grid Search** sob validação cruzada de 10 dobras estratificadas.

**O Racional Técnico — O Ponto Chave.**
A finalização usa `last_fit` sobre o *split* original, avaliando o modelo em um **conjunto de teste inédito (20%)** — protocolo correto que impede contaminação. Extraímos não só Acurácia e AUC, mas também **Sensibilidade e Especificidade com `event_level = "second"`**, e usamos `vip` para revelar quais fatores clínicos mais pesam no diagnóstico, transformando o modelo em **insight interpretável** para a área da saúde.

```r
# Curadoria de 15 variáveis clínicas + receita à prova de vazamento
data_recipe <- recipe(gallstone_status ~ c_reactive_protein_crp + vitamin_d +
    aspartat_aminotransferaz_ast + high_density_lipoprotein_hdl + ... ,
    data = train_data) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.90)

# Avaliação honesta: modelo final julgado em teste inédito
final_fit <- last_fit(final_wf, split = data_split)
```

---

### 💊 Rodada 3 de Competição (30/03) — Base Drogas (18 Substâncias)

📂 [`ConsumoDrogas2/`](ConsumoDrogas2/) · Scripts: [`Consumo2.R`](ConsumoDrogas2/Consumo2.R) · [`SemSMOTE.R`](ConsumoDrogas2/SemSMOTE.R)

**O Desafio.** Prever o consumo de **18 substâncias diferentes** a partir de traços de personalidade (Big Five), impulsividade e *sensation seeking*. Cada substância é um problema de classificação independente, com distribuições de classe radicalmente diferentes entre si.

**A Abordagem.**
- **Pipeline iterativo:** um único *loop* percorre as 18 drogas, e para cada uma promovemos uma **competição interna entre três algoritmos** — **Ranger** (Random Forest), **Glmnet** (Elastic Net) e **KNN** (`kknn`) — declarando campeão aquele com maior Acurácia (desempate por F1).
- **Normalização obrigatória:** todos os preditores numéricos passam por `step_normalize`, condição indispensável para o bom funcionamento de Glmnet e KNN, sensíveis à escala.

**O Racional Técnico — O Ponto Chave.**
Aplicamos o **SMOTE de forma condicional**: o oversampling sintético só é acionado quando a classe minoritária representa **menos de 30%** do treino. Para *provar matematicamente* que essa decisão era necessária — e não um capricho —, mantivemos um **script espelho sem SMOTE** ([`SemSMOTE.R`](ConsumoDrogas2/SemSMOTE.R)). A comparação direta entre os dois cenários demonstra empiricamente o ganho de Sensibilidade e F1 trazido pelo balanceamento nas substâncias raras.

```r
# SMOTE acionado apenas quando a minoria é < 30%
rec <- recipe(Classe ~ ., data = treino_raw) %>%
  step_normalize(all_numeric_predictors())

if (!is.na(prop_usuarios) && prop_usuarios < 0.30) {
  rec <- rec %>% step_smote(Classe)   # oversampling sintético sob demanda
}
```

---

### 🧪 Rodada 4 de Competição (06/03) — Base InterDia (Descritores Moleculares RDKit)

📂 [`Dia/`](Dia/) · Script: [`Codigo.R`](Dia/Codigo.R) · Relatório: [`relatorio_DIA (1) (1).pdf`](Dia/relatorio_DIA%20(1)%20(1).pdf)

**O Desafio.** Classificação binária sobre **descritores moleculares RDKit** de alta dimensionalidade, com a meta de atingir a **acurácia absoluta máxima** — o estudo mais "pesado" do conjunto.

**A Abordagem.**
- **Arquitetura de Ensemble Stacking** construída com o ecossistema **`tidymodels` + `stacks`**: três modelos-base de naturezas distintas — **XGBoost**, **Random Forest (Ranger)** e **SVM Radial (kernlab)** — todos com hiperparâmetros otimizados via `tune_grid`.
- **Meta-modelo:** as previsões dos três são combinadas por uma **Regressão Lasso** (`blend_predictions`), que aprende os pesos ótimos de cada modelo-base, descartando os redundantes.

**O Racional Técnico — O Ponto Chave.**
Dois cuidados garantiram a integridade e o pico de desempenho:
1. **Isolamento do SMOTE contra Data Leakage:** o `step_smote` está embutido na `recipe`, o que faz o balanceamento ser recalculado **dentro de cada dobra de treino** da validação cruzada de 10 folds — nunca tocando os dados de validação. Sem isso, a acurágia reportada seria uma ilusão otimista.
2. **Varredura de força bruta no Threshold:** após o ensemble, testamos **todos os limiares de `0.01` a `0.99`** sobre as probabilidades brutas, localizando o ponto de corte que entrega o **cume da acurácia**.

```r
# Receita à prova de vazamento: SMOTE recalculado por fold
dia_recipe <- recipe(Label ~ ., data = train_data) %>%
  step_nzv(all_predictors()) %>%
  step_corr(all_predictors(), threshold = 0.90) %>%
  step_normalize(all_predictors()) %>%
  step_smote(Label, over_ratio = 1)   # apenas no treino de cada dobra

# Força bruta no limiar para acurácia máxima
limiares <- seq(0.01, 0.99, by = 0.01)
resultados_acc <- sapply(limiares, function(t) {
  pred <- factor(ifelse(test_preds$.pred_X1 >= t, "X1", "X0"), levels = c("X1", "X0"))
  yardstick::accuracy_vec(truth = test_preds$Label, estimate = pred)
})
melhor_limiar_acc <- limiares[which.max(resultados_acc)]
```

---

### 🩺 Rodada 5 de Competição (13/03) — Base ILPD (Pacientes Hepáticos)

📂 [`ILPD/`](ILPD/) · Script: [`Codigo.R`](ILPD/Codigo.R) · Relatório: [`relatorio_ILPD_ARS.pdf`](ILPD/relatorio_ILPD_ARS.pdf)

**O Desafio.** Diagnosticar doença hepática no ***Indian Liver Patient Dataset*** — uma base **assimétrica**, com **vieses biológicos** (diferenças por sexo), outliers e exames em escalas muito distintas.

**A Abordagem.**
- **Injeção de domínio clínico via Feature Engineering:** criamos a **Razão AST/ALT** (`SGOT / SGPT`, o clássico *De Ritis ratio*, marcador de padrão de lesão hepática) e a **Razão de Bilirrubinas** (`DB / TB`, bilirrubina direta sobre total) — variáveis que um hepatologista olharia primeiro.
- **Transformação Yeo-Johnson:** aplicada com `center` e `scale` para **estabilizar a variância**, domar os outliers dos exames bioquímicos e aproximar as distribuições da normalidade (a Yeo-Johnson lida com valores zero/negativos, onde a Box-Cox falharia).

**O Racional Técnico — O Ponto Chave.**
Optamos pelo **SVM com kernel Radial**, escolha deliberada: o SVM é **diretamente beneficiado pela centralização e padronização** trazidas pela Yeo-Johnson, pois opera com distâncias no espaço de atributos. O modelo final foi **polido por um *threshold tuning* minucioso** (varredura de `0.01` a `0.99`), extraindo o máximo de acurácia possível da base desbalanceada.

```r
# Domínio clínico transformado em atributo
df$AST_ALT_Ratio <- df$SGOT / df$SGPT   # De Ritis ratio
df$Bili_Ratio    <- df$DB   / df$TB     # bilirrubina direta / total

# Yeo-Johnson estabiliza variância e outliers — terreno ideal para o SVM Radial
pre_proc <- preProcess(df_train, method = c("YeoJohnson", "center", "scale"))
```

---

### 🧀 Base de dados Extra - 06 — Segmentação Sensorial de Queijos (Análise Não-Supervisionada)

📂 [`Cheese/`](Cheese/) · Script: [`ScriptExtra.R`](Cheese/ScriptExtra.R) · Base: [`cheese.xls`](Cheese/cheese.xls) · Relatório: [`relatorio_Cheese.pdf`](Cheese/relatorio_Cheese.pdf)

**O Desafio.** Diferentemente dos demais, este estudo **não possui variável-alvo**. O objetivo é puramente exploratório: descobrir **agrupamentos naturais (clusters)** em dados de avaliação **sensorial** de queijos, identificando perfis de produto que não estão rotulados nos dados.

**A Abordagem.**
- **Seleção e saneamento:** isolamento das colunas sensoriais, conversão robusta para numérico e descarte de variáveis com excesso de ausentes (> 50%), seguido de **padronização** (`scale`) — passo indispensável para algoritmos baseados em distância.
- **Definição do número de clusters:** aplicação do **Método do Cotovelo** (*Elbow Method*, via `fviz_nbclust` com soma dos quadrados intra-cluster) para fundamentar a escolha de **k = 3**.
- **Clusterização:** **K-Means** com `nstart = 25` (25 inicializações aleatórias para fugir de mínimos locais).
- **Interpretação:** redução de dimensionalidade por **PCA** (`prcomp`) para visualizar a separação dos grupos, análise de **contribuição das variáveis** e **heatmap de perfil** dos clusters.

**O Racional Técnico — O Ponto Chave.**
O valor deste estudo está em demonstrar **domínio do aprendizado não-supervisionado**: não basta rodar o K-Means, é preciso **justificar o número de clusters** (cotovelo), **garantir comparabilidade entre variáveis** (padronização obrigatória) e, sobretudo, **traduzir os clusters em conhecimento** — o cruzamento entre PCA e o heatmap de médias revela *quais atributos sensoriais definem cada perfil de queijo*.

```r
# Padronização → cotovelo → K-Means → interpretação por PCA
dados_scaled <- scale(dados_num)
fviz_nbclust(dados_scaled, kmeans, method = "wss")          # define k
kmeans_result <- kmeans(dados_scaled, centers = 3, nstart = 25)
fviz_cluster(kmeans_result, data = dados_scaled, ellipse.type = "convex")
```

**Resultados.** Sobre **240 avaliações sensoriais**, o K-Means produziu três grupos equilibrados (84 / 75 / 81 amostras), que a leitura dos perfis médios traduziu em **três identidades de queijo nitidamente distintas**:

| Cluster | Perfil | Assinatura sensorial |
|---|---|---|
| **1** (84) | 🧈 *Cremoso e Amanteigado* | alta cremosidade, derretimento e brilho; amanteigado e salgado; baixa firmeza |
| **2** (75) | 🍋 *Ácido e Magro* | acidez dominante e textura arenosa; baixo teor de gordura, manteiga e sal |
| **3** (81) | 🧱 *Firme e Gorduroso* | alta firmeza e resistência à mordida; muito gorduroso; pouco derretimento e opaco |

> Detalhamento completo (cotovelo, mapa de clusters via PCA, contribuição das variáveis e heatmap de perfis) no relatório técnico: [`relatorio_Cheese.pdf`](Cheese/relatorio_Cheese.pdf).

---

## ⚙️ Como Reproduzir os Códigos

### Pré-requisitos

- **R** ≥ 4.1 (recomendado **RStudio**)
- Conexão à internet na primeira execução (os scripts **instalam automaticamente** os pacotes ausentes)

### Pacotes por estudo

| Estudo | Pacotes principais |
|---|---|
| **Rodada 1 — Cartão de Crédito** | `readxl`, `caret`, `randomForest`, `pROC`, `dplyr`, `smotefamily`, `doParallel` |
| **Rodada 2 — Gallstone (Vesícula)** | `tidymodels`, `readxl`, `dplyr`, `ranger`, `vip`, `ggplot2`, `janitor` |
| **Rodada 3 — Drogas** | `dplyr`, `caret`, `themis`, `recipes`, `ranger`, `glmnet`, `kknn`, `ggplot2`, `tidyr`, `knitr` |
| **Rodada 4 — InterDia** | `tidymodels`, `themis`, `xgboost`, `ranger`, `kernlab`, `stacks`, `finetune`, `doParallel`, `vip`, `ggplot2` |
| **Rodada 5 — ILPD** | `tidyverse`, `caret`, `kernlab`, `pROC` |
| **Extra 06 — Queijos (Cheese)** | `readxl`, `ggplot2`, `factoextra`, `dplyr`, `reshape2` |

### Execução

Cada estudo é um projeto RStudio (`.Rproj`) independente e autocontido. Para rodar qualquer um deles:

```r
# 1. Abra o .Rproj da pasta desejada (ex.: CartaoCredito/CartaoCredito.Rproj)
#    Isso garante que o diretório de trabalho aponte para os dados corretos.

# 2. Execute o script na íntegra (Ctrl+Shift+Enter no RStudio)
source("cartaoCreditoARS.R")
```

Os scripts cuidam de **instalação de dependências, carga dos dados, pré-processamento, treino, avaliação e geração de gráficos** automaticamente. As métricas finais (matrizes de confusão, acurácia, AUC e limiares ótimos) são impressas no console, e os gráficos aparecem no painel *Plots*.

> ⚠️ **Nota sobre os dados.** Cada pasta espera o respectivo arquivo de dados em seu diretório de trabalho. Caso algum dataset bruto (ex.: `drug_consumption.data`) não esteja presente, baixe-o da fonte original (UCI Machine Learning Repository) e posicione-o na pasta correspondente antes de executar.

---

## 🏁 Conclusão

Mais do que perseguir um número de acurácia, este trabalho consolidou um **método**. Em domínios completamente diferentes — finanças, comportamento, química, medicina e até análise sensorial de alimentos — a mesma disciplina técnica se repetiu e se mostrou decisiva:

- ✅ **A Feature Engineering vence o algoritmo.** Variáveis criadas com intenção (scores comportamentais, razões clínicas) entregaram mais ganho que qualquer troca de modelo.
- ✅ **Data Leakage é um inimigo silencioso.** Isolar SMOTE e normalização dentro das dobras de treino foi o que separou métricas *honestas* de métricas *infladas*.
- ✅ **O limiar de `0.50` raramente é o melhor.** O *threshold tuning* — condicional ou por força bruta — extraiu desempenho que estava, literalmente, parado na mesa.
- ✅ **Cada base pede sua própria arquitetura.** De Random Forest com SMOTE a Ensemble Stacking, de Elastic Net a SVM Radial sobre Yeo-Johnson: não há bala de prata, há adequação.
- ✅ **Supervisionado e não-supervisionado.** Da classificação de risco e diagnóstico à segmentação por K-Means com PCA, o repositório cobre os dois grandes paradigmas da Mineração de Dados com o mesmo rigor de pré-processamento e interpretação.

O resultado é um conjunto de pipelines reprodutíveis, justificados linha a linha, que demonstram **rigor metodológico** e **maturidade técnica** na prática da Mineração de Dados.

---

<p align="center">
  <em>Desenvolvido por Arthur Faria, Ricardo Issa e Sávio Issa</em><br>
  <strong>IF Goiano — Campus Urutaí · Sistemas de Informação</strong>
</p>
