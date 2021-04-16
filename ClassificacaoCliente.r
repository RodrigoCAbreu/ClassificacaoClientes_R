# Classificação Multiclasse com SVM - Prevendo Gastos com Cartão de Crédito em 3 Categorias

####  Definido o Problema de Negócio #### 

# A identificação e a capacidade de classificar os clientes com base nos gastos sempre foram uma área de 
# interesse para instituições bancárias e empresas de cartão de crédito. É um aspecto importante no 
# gerenciamento de relacionamento com o cliente e ajuda a aumentar a receita com clientes existentes. Várias 
# tentativas foram feitas a esse respeito. Os emissores de cartões de crédito tradicionalmente têm como alvo 
# os consumidores usando informações sobre seus comportamentos e dados demográficos. 

# Nosso trabalho é classificar os clientes de cartão de crédito de acordo com seu comportamento de gastos. 
# A segmentação é um aspecto importante na compreensão do cliente e na execução de campanhas de marketing 
# eficazes e rentáveis. Usaremos o SVM como nosso modelo.

# Os dados demográficos, os detalhes sobre emprego e o estilo de vida dos clientes desempenham um papel vital na 
# maneira como eles gastam. Existem fatores ocultos, bem como semelhança com as compras. A máquina de vetores 
# de suporte pode ser usada para problemas de regressão e classificação. 

# Usaremos SVM com Kernel Linear Multiclasse como nosso modelo proposto para classificar a variável target. 
# No entanto, também avaliaremos outros Kernels, como RBF e Polinomial, para uma variedade de hiperparâmetros. 
# Também levamos em consideração o viés no dados.

# Fonte dos dados: https://sorry.vse.cz/~berka/ (dados anônimos)

# Pacotes
install.packages("gains")
install.packages("pROC")
install.packages("ROSE")
install.packages("mice")
install.packages("xts")
library(dplyr)
library(caret)
library(gains)
library(pROC)
library(ROCR)
library(ROSE)
library(e1071)
library(mice)

# Carregando os dados
dataset_clientes <- read.csv("dados/cartoes_clientes.csv")
View(dataset_clientes)


#### Análise Exploratória dos Dados #### 
str(dataset_clientes)
summary(dataset_clientes)
summary(dataset_clientes$card2spent)

# Removemos a variável com ID do cliente pois não é necessário
dataset_clientes <- dataset_clientes[-1]
View(dataset_clientes)

# Checando valores missing
sapply(dataset_clientes, function(x)sum(is.na(x)))

# Checando se a variável alvo está balanceada
table(dataset_clientes$Customer_cat)
prop.table(table(dataset_clientes$Customer_cat)) * 100

# Outra alternativa
as.data.frame(table(dataset_clientes$Customer_cat))

# Análise Visual

# BoxPlot e Histograma
boxplot(dataset_clientes$card2spent)
summary(dataset_clientes$card2spent)
hist(dataset_clientes$card2spent)

boxplot(dataset_clientes$hourstv)
summary(dataset_clientes$hourstv)
hist(dataset_clientes$hourstv)

# Scatter Plot
plot(dataset_clientes$card2spent, dataset_clientes$hourstv, xlab = "Gasto Cartão", ylab = "Horas TV")


# Função para Fatorização de variáveis categóricas
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(paste(df[[variable]]))
  }
  return(df)
}

# Lista de varáveis categóricas
categorical.vars <- c('townsize', 'jobcat', 'retire', 'hometype', 'addresscat', 
                      'cartype', 'carvalue', 'carbought', 'card2', 'gender', 'card2type', 
                      'card2benefit', 'card2benefit', 'bfast', 'internet', 'Customer_cat')

# Fatorizando as variáveis categóricas (alterando as variáveis categóricas para fatores)
str(dataset_clientes)
dataset_clientes <- to.factors(df = dataset_clientes, variables = categorical.vars)
str(dataset_clientes)
View(dataset_clientes)
str(dataset_clientes$gender)


# Aplicando Imputação em Valores Missing Usando Método PMM (Predictive Mean Matching)

# Checando valores missing
sapply(dataset_clientes, function(x)sum(is.na(x)))
sum(is.na(dataset_clientes))


# Descobrindo os números das colunas das variáveis fatores, para excluí-las da imputação
fac_col <- as.integer(0)
facnames <- names(Filter(is.factor, dataset_clientes))
k = 1

for(i in facnames){
  while (k <= 16){
    grep(i, colnames(dataset_clientes))
    fac_col[k] <- grep(i, colnames(dataset_clientes))
    k = k + 1
    break
  }
}

# Colunas que são do tipo fator
fac_col

# Imputação

# Fatiamento do dataset
View(dataset_clientes)
View(dataset_clientes[,-c(fac_col)])

# Definindo a regra de imputação
?mice
regra_imputacao <- mice((dataset_clientes[,-c(fac_col)]), 
                        m = 1, 
                        maxit = 50, 
                        meth = 'pmm', 
                        seed = 500)

# Aplicando a regra de imputação
?mice::complete
total_data <- complete(regra_imputacao, 1)
View(total_data)

# Junta novamente as variáveis categóricas
dataset_clientes_final <- cbind(total_data, dataset_clientes[,c(fac_col)])
View(dataset_clientes_final)

# Dimensões
dim(dataset_clientes_final)

# Tipos de dados
str(dataset_clientes_final)
str(dataset_clientes_final$gender)

# Checando valores missing
sapply(dataset_clientes_final, function(x)sum(is.na(x)))
sum(is.na(dataset_clientes_final))
sum(is.na(dataset_clientes))

# Variável target como fator
dataset_clientes_final$Customer_cat <- as.factor(dataset_clientes_final$Customer_cat)
str(dataset_clientes_final$Customer_cat)

# Dividindo randomicamente o dataset em 80% para dados de treino e 20% para dados de teste

# Seed para reproduzir os mesmos resultados
set.seed(100)

# Índice de divisão dos dados
indice_divide_dados <- sample(x = nrow(dataset_clientes_final),
                              size = 0.8 * nrow(dataset_clientes_final),
                              replace = FALSE)
View(indice_divide_dados)

# Aplicando o índice
dados_treino <- dataset_clientes_final[indice_divide_dados,]
dados_teste <- dataset_clientes_final[-indice_divide_dados,]

View(dados_treino)
View(dados_teste)

# Checando o balanceamento de classe da variável target
prop.table(table(dados_treino$Customer_cat)) * 100

# Podemos ver que os dados apresentam um desequilíbrio alto com:
# 2% high_spend_cust, 30% low_spend_cust enquanto a maioria de 68% é medium_spent_cust
# Vamos balancear a classe usando Oversampling com SMOTE.

# Balanceamento de Classe com SMOTE
# Oversampling x Undersampling

# Seed
set.seed(301)

# Pacote
install.packages("DMwR")
library(DMwR)
??DMwR

# SMOTE - Synthetic Minority Oversampling Technique
?SMOTE
dados_treino_balanceados <- SMOTE(Customer_cat ~ ., dados_treino, perc.over = 3000, perc.under = 200)

# Checando o balanceamento de classe da variável target
prop.table(table(dados_treino_balanceados$Customer_cat)) * 100

# Salvando os datasets após o pré-processamento
class(dados_treino_balanceados)
class(dados_teste)

write.csv(dados_treino_balanceados, "dados/dados_treino_balanceados.csv")
write.csv(dados_teste, "dados/dados_teste.csv")

dim(dados_treino_balanceados)
dim(dados_teste)

View(dados_treino_balanceados)
View(dados_teste)

sum(is.na(dados_treino_balanceados))
sum(is.na(dados_teste))
sapply(dados_teste, function(x)sum(is.na(x)))

# Carregando os dados pré-processados
?read.csv
dados_treino1 <- read.csv("dados/dados_treino_balanceados.csv")
dados_teste1 <- read.csv("dados/dados_teste.csv")
dim(dados_treino1)
dim(dados_teste1)
View(dados_treino1)
View(dados_teste1)

# A função read_csv mostra o que aconteceu
?read_csv
dados_treino <- read_csv("dados/dados_treino_balanceados.csv")
dados_teste <- read_csv("dados/dados_teste.csv")
dim(dados_treino)
dim(dados_teste)
View(dados_treino)
View(dados_teste)

# Removemos a coluna X criada na indexação randômica
dados_treino <- dados_treino[-1]
dados_teste <- dados_teste[-1]

dim(dados_treino)
dim(dados_teste)

# Transformando a variável target em valor numérico
View(dados_treino)
str(dados_treino$Customer_cat)
View(dados_treino$Customer_cat)
dados_treino$Customer_cat <- as.numeric(as.factor(dados_treino$Customer_cat))
str(dados_treino$Customer_cat)
View(dados_treino$Customer_cat)
dados_treino$Customer_cat <- as.factor(dados_treino$Customer_cat)
str(dados_treino$Customer_cat)
View(dados_treino$Customer_cat)

dados_teste$Customer_cat <- as.numeric(as.factor(dados_teste$Customer_cat))
dados_teste$Customer_cat <- as.factor(dados_teste$Customer_cat)
str(dados_teste$Customer_cat)
View(dados_teste$Customer_cat)


##### Modelagem Preditiva ##### 

# Primeira versão do modelo SVM - Versão Padrão com Kernel Radial (RBF)
# O algoritmo escolhe o tipo de SVM de acordo com o tipo de dado da variável target
?svm
modelo_v1 <- svm(Customer_cat ~ ., data = dados_treino, na.action = na.omit, scale = TRUE)
summary(modelo_v1)
print(modelo_v1)

# Fazendo previsões com o modelo
previsoes_v1 <- predict(modelo_v1, newdata = dados_teste)

# Matriz de Confusão
?caret::confusionMatrix
caret::confusionMatrix(previsoes_v1, dados_teste$Customer_cat)

# Por que o erro aconteceu? Vamos checar! Isso chama-se troubleshooting.

# Comprimento do valor real e do valor previsto
length(dados_teste$Customer_cat)
length(previsoes_v1)

# Temos valores NA em teste?
sum(is.na(dados_teste))

# Removemos valores NA
dados_teste = na.omit(dados_teste)
length(dados_teste$Customer_cat)
sum(is.na(dados_teste))

# E agora sim a matriz de confusão
caret::confusionMatrix(previsoes_v1, dados_teste$Customer_cat)

# Métricas
install.packages("multiROC")
library(multiROC)

?multiclass.roc
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = previsoes_v1)
class(dados_teste$Customer_cat)
class(previsoes_v1)

# Faz a conversão
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v1)))

# Score AUC (Area Under The Curve)
curva_roc$auc

# Juntando valores reais e previstos na mesma tabela

# Previsões
valores_previstos <- data.frame(as.numeric(as.factor(previsoes_v1)))
colnames(valores_previstos) <- ("previsão")

# Valores reais
valores_reais <- data.frame(as.numeric(as.factor(dados_teste$Customer_cat)))
colnames(valores_reais) <- ("valor_real")

# Dataframe final
final_df <- cbind(valores_reais, valores_previstos)
View(final_df)


# Segunda versão do modelo SVM - Versão com Kernel Linear e GridSearch

# Vamos fazer uma pesquisa em grade (Grid Search) para o ajuste de hiperparâmetros e usar Kernel linear. 
# Mas aqui não manteremos o custo superior a 2, para que valores discrepantes não afetem extensivamente 
# a criação de limites de decisão e, portanto, levem ao ajuste excessivo (overfitting).
set.seed(182)
?tune
modelo_grid1 <- tune(svm, 
                     Customer_cat ~ ., 
                     data = dados_treino, 
                     kernel = 'linear',
                     ranges = list(cost = c(0.05, 0.1, 0.5, 1, 2))) 


summary(modelo_grid1)

# Parâmetros do melhor modelo
modelo_grid1$best.parameters

# Melhor modelo
modelo_grid1$best.model 
modelo_v2 <- modelo_grid1$best.model 
summary(modelo_v2)

# Previsões
previsoes_v2 <- predict(modelo_v2, dados_teste)

# Matriz de Confusão e Score AUC
confusionMatrix(previsoes_v2, dados_teste$Customer_cat)
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v2)))
curva_roc$auc

# Está oferecendo um desempenho muito melhor em termos de sensibilidade na previsão de classes minoritárias


# Terceira versão do modelo SVM - Versão com Kernel RBF e Otimização no Parâmetro Gamma

# Vamos fazer uma pesquisa em grade para o ajuste de parâmetros com kernel radial, e não manteremos 
# o custo superior a 2, para que os discrepantes não afetem extensivamente a criação de limites 
# de decisão e, portanto, levem ao ajuste excessivo.

# Da mesma forma, não manteremos um valor muito abaixo de 0,001 para gamma, pois isso levaria a 
# um excesso de ajuste.
set.seed(182)
modelo_grid2 <- tune(svm,
                     Customer_cat ~ .,
                     data = dados_treino,
                     kernel='radial',
                     ranges = list(cost = c(0.01,0.05,0.1,0.5,1,2),
                                   gamma = c(0.0001,0.001,0.01,.05,0.1,0.01,1,2)))

summary(modelo_grid2)

# Parâmetros do melhor modelo
modelo_grid2$best.parameters

# Melhor modelo
modelo_v3 <- modelo_grid2$best.model 

# Previsões
previsoes_v3 <- predict(modelo_v3, dados_teste)

# Matriz de Confusão e Score AUC
confusionMatrix(previsoes_v3, dados_teste$Customer_cat)
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v3)))
curva_roc$auc


# Quarta versão do modelo SVM - Versão com Kernel Polinomial

# Vamos fazer uma pesquisa em grade para ajustar parâmetros com kernel poliinomial, e não manteremos 
# o custo superior a 2, para que os discrepantes não afetem extensivamente a criação de limites 
# de decisão e, portanto, levem a um excesso de ajuste.

# Da mesma forma, não manteremos o grau polinomial de ordem superior a 4, pois isso levaria a 
# um ajuste excessivo
set.seed(182)
modelo_grid3 <- tune(svm,
                     Customer_cat ~ .,
                     data = dados_treino,
                     kernel = 'polynomial',
                     ranges = list(cost = c(1,2), degree = c(2,3,4)))

summary(modelo_grid3)

# Parâmetros do melhor modelo
modelo_grid3$best.parameters

# Melhor modelo
modelo_v4 <- modelo_grid3$best.model 

# Previsões
previsoes_v4 <- predict(modelo_v4, dados_teste)

# Matriz de Confusão e Score AUC
confusionMatrix(previsoes_v4, dados_teste$Customer_cat)
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v4)))
curva_roc$auc

# Podemos ver que o modelo ajustado com kernel polinomial tem uma sensibilidade fraca na 
# previsão de clientes com alto gasto para o conjunto de testes.


# Comparação dos Modelos

# Resultados do Modelo 1
resultados_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$Customer_cat)
resultados_v1$overall
resultados_v1$byClass

# Medidas Globais do Modelo 1
acuracia_v1 <- resultados_v1$overall['Accuracy']
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v1)))
score_auc_v1 <- curva_roc$auc

# Exemplo: Caso você queira outras medidas como precision e recall, lembre-se que elas são por classe
precision_v1_classe1 <- resultados_v1$byClass[1, 'Precision']   
precision_v1_classe2 <- resultados_v1$byClass[2, 'Precision']  
recall_v1_classe3 <- resultados_v1$byClass[3, 'Sensitivity']

# Vetor com os resultados de avaliação do Modelo v1
vetor_modelo1 <- c("Modelo1 Kernel RBF", round(acuracia_v1, 4), round(score_auc_v1, 4))


# Medidas Globais do Modelo 2
resultados_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$Customer_cat)
acuracia_v2 <- resultados_v2$overall['Accuracy']
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v2)))
score_auc_v2 <- curva_roc$auc

# Vetor com os resultados de avaliação do Modelo v2
vetor_modelo2 <- c("Modelo2 Kernel Linear", round(acuracia_v2, 4), round(score_auc_v2, 4))


# Medidas Globais do Modelo 3
resultados_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$Customer_cat)
acuracia_v3 <- resultados_v3$overall['Accuracy']
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v3)))
score_auc_v3 <- curva_roc$auc

# Vetor com os resultados de avaliação do Modelo v1
vetor_modelo3 <- c("Modelo3 Kernel RBF Tunning", round(acuracia_v3, 4), round(score_auc_v3, 4))


# Medidas Globais do Modelo 4
resultados_v4 <- caret::confusionMatrix(previsoes_v4, dados_teste$Customer_cat)
acuracia_v4 <- resultados_v4$overall['Accuracy']
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v4)))
score_auc_v4 <- curva_roc$auc

# Vetor com os resultados de avaliação do Modelo v1
vetor_modelo4 <- c("Modelo4 Kernel Polinomial", round(acuracia_v4, 4), round(score_auc_v4, 4))


# Concatenando os resultados
# Dataframe para os resultados dos modelos
?base::rbind
compara_modelos <- rbind(vetor_modelo1, vetor_modelo2, vetor_modelo3, vetor_modelo4)
View(compara_modelos)
rownames(compara_modelos) <- c("1", "2", "3", "4")
colnames(compara_modelos) <- c("Modelo", "Acuracia", "AUC")
View(compara_modelos)
class(compara_modelos)
compara_modelos <- as.data.frame(compara_modelos)
class(compara_modelos)
View(compara_modelos)

# Plot
library(ggplot2)

# Acurácia
ggplot(compara_modelos, aes(x = Modelo, y = Acuracia, fill = Modelo)) + 
  geom_bar(stat = "identity") 

# AUC
ggplot(compara_modelos, aes(x = Modelo, y = AUC, fill = Modelo)) + 
  geom_bar(stat = "identity")

# Assim, o método final proposto é baseado no Kernel Linear, a versão 2 do nosso modelo.


# Previsões com o Modelo Escolhido

# Salvando o modelo selecionado
?saveRDS
saveRDS(modelo_v2, "modelos/modelo_v2.rds")

# Carregando o modelo salvo
modelo_svm <- readRDS("modelos/modelo_v2.rds")
print(modelo_svm)

# Carrega o arquivo com dados de novos clientes.
# Para esses clientes não temos a variável target, pois isso é o que queremos prever.
novos_clientes <- read.csv("dados/novos_clientes.csv", header = TRUE)
View(novos_clientes)
dim(novos_clientes)

# Fazendo previsões
previsoes_novos_clientes <- predict(modelo_svm, novos_clientes)

# Apresentando o resultado final

# Previsões
previsoes_gastos_novos_clientes <- data.frame(as.numeric(as.factor(previsoes_novos_clientes)))
colnames(previsoes_gastos_novos_clientes) <- ("Previsão de Gasto")

# Idade dos clientes
idades_novos_clientes <- data.frame(novos_clientes$age)
colnames(idades_novos_clientes) <- ("Idades")

# Dataframe final
resultado_final <- cbind(idades_novos_clientes, previsoes_gastos_novos_clientes)
View(resultado_final)

# Ajusta o label da previsão
library(plyr)
?mapvalues
resultado_final$`Previsão de Gasto` <- mapvalues(resultado_final$`Previsão de Gasto`,
                                                 from = c(1,2,3),
                                                 to = c("Alto", "Médio", "Baixo"))

View(resultado_final)
write.csv(resultado_final, "dados/resultado_final.csv")







