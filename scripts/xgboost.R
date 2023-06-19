
library(quanteda)
library(tidyverse)
library(feather)
library(xgboost)

# Cargar archivo editado
editado <- read_feather("data/edicion_inicial.feather")

editado2 <- editado %>% 
  filter(!is.na(polo)) %>%
  filter(polo != "centro") %>% 
  filter(anio >= 1990) %>% 
  sample_frac(0.1) %>% 
  select(text = texto_dep, polo) %>% 
  mutate(polo_fct = as.numeric(as.factor(polo)) - 1,
         polo_fct2 = if_else(polo_fct == 1, 0, 1)
         )
  
  
  stopwords <- c(quanteda::stopwords("es"), "señor", "señores", "proyecto", "presidente", "presidenta", "dicho", "honorable", "don",
                 "nyo", "garcia", "sanfuentes", "bulnes", "ibañez", "arnello", "ochagavia", "phillips", 
                 "von", "muhlenbrock", "gustavo", "mühlenbrock", "nel", "solo", "sólo", "n°", "vieragallo"
  )



  dfm <- editado2  %>% 
    corpus() %>% 
    tokens(remove_punct = T, remove_symbols = T, remove_numbers = T) %>%
    tokens_select(pattern = stopwords, selection = "remove") %>%
    tokens_select(min_nchar = 2, selection = "remove") %>% 
    dfm() 
  

tfidf <- dfm %>% 
  dfm_trim(min_termfreq = 20) %>% 
  dfm_tfidf()

tfidf %>% dim()

n <- 8000
x_data <- as.matrix(tfidf[1:n, ])
y = tfidf@docvars$polo_fct2[1:n]
xgb_train = xgb.DMatrix(data = x_data, label = y )
model = xgboost(data = xgb_train, max.depth = 6, nrounds = 70,  nthread = 10, verbose = 1 )


pred <- predict(model, x_data)
err <- mean(as.numeric(pred > 0.5) == y)
err

importance_matrix = xgb.importance(colnames(xgb_train), model = model)
importance_matrix
juiiiiiiiii

xgb.importance(model = model, trees = seq(from=0, by=2, length.out = 6))
xgb.importance(model = model, trees = seq(from=1, by=2, length.out=6))
