## libraries
library(tidyverse)
library(here)
library(stringr)
library(xgboost)
library(parsnip)

## Function

cabins<- function(Cabin){
  len<- c()
  cab<- ifelse(Cabin == "", NA, str_split(Cabin, " "))
  for (i in 1:length(cab)){
    if (!is.na(cab[i])) {
      len<- c(len,length(cab[i]))}
    else {len<- c(len,0)}
  }
  len
}
#cabins(train_df$Cabin) %>% head()

## Data
data_col<- c('Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare','Embarked','Deck','num_cab')

train_df<- read.csv(here("train.csv")) %>%
  mutate(Survived = ifelse(Survived == 1, TRUE, FALSE),
         Sex = as.numeric(factor(Sex)),
         Embarked = ifelse(Embarked == "", NA, Embarked),
         Embarked = as.numeric(factor(Embarked, levels = c("C", "Q", "S"))),
         Deck = str_extract(Cabin, "[A,B,C,D,E]"),
         Deck = ifelse(is.na(Deck), "none", Deck),
         Deck = as.numeric(factor(Deck)),
         Pclass = factor(Pclass),
         last = str_extract(Name, "[a-zA-Z]+"),
         num_cab = cabins(Cabin),
         Survived = as.numeric(Survived),
         Pclass = as.numeric(Pclass)
  ) %>%
  select(all_of(data_col))

species<- train_df$Survived %>% factor()

df<- train_df %>% select(-c("Survived")) %>% as.matrix()
label<- train_df %>% select("Survived")

## split
set.seed(467)
rows<- nrow(train_df)
split<- sample(seq(1, 3), size = rows, replace = TRUE, prob = c(.8, .2, .2))

train<- xgb.DMatrix(data=df[split == 1,], label = label[split == 1,])
valid<- df[split == 2,]%>% xgb.DMatrix(label = label[split == 2,])
test<- df[split == 3, ]%>% xgb.DMatrix(label = label[split == 3,])

train_lab<- label[split == 1,]
valid_lab<- label[split == 2,]
test_lab<- label[split == 3, ]


xgb.fit= xgboost(data = train, max.depth = 9, eta=0.01, nthread = 4, 
                nrounds = 10000, objective = "binary:logistic",
                eval_metric = "error", early_stopping_rounds = 100,
                colsample_bytree = 0.5, subsample = 0.5, num_parallel_tree = 10, 
                booster="gbtree")

xgb.pred = predict(xgb.fit,valid,reshape=T)
mean(as.numeric(xgb.pred>0.5)== valid_lab)


params <- list(
  booster="gbtree",
  eta=0.001,
  max_depth=9,
  subsample=0.50,
  colsample_bytree=0.5,
  num_parallel_tree = 10,
  objective="binary:logistic",
  eval_metric="error",
  lambda = 3
)

md2<- xgb.train(params = params, data = train, nrounds = 10000, nthreads=4, 
                early_stopping_rounds = 100, 
                watchlist = list(train=train, val = valid), verbose = 1) 

xgb.pred<- predict(md2, valid, reshape = T)  
mean(as.numeric(xgb.pred>0.50)== valid_lab)


