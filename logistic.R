## libraries
library(tidyverse)
library(here)
library(stringr)
library(xgboost)

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
train_df<- read.csv(here("train.csv")) %>%
  mutate(Survived = ifelse(Survived == 1, TRUE, FALSE),
         Sex = factor(Sex),
         Embarked = ifelse(Embarked == "", NA, Embarked),
         Embarked = factor(Embarked, levels = c("C", "Q", "S"),
                           labels = c("Cherbourg", 
                                      "Queenstown", "Southampton")),
         Deck = str_extract(Cabin, "[A,B,C,D,E]"),
         Deck = ifelse(is.na(Deck), "none", Deck),
         Pclass = factor(Pclass),
         last = str_extract(Name, "[a-zA-Z]+"),
         num_cab = cabins(Cabin)
         )

## split
set.seed(467)
rows<- nrow(train_df)
split<- sample(seq(1, 3), size = rows, replace = TRUE, prob = c(.8, .2, .2))

train<- train_df[split == 1,]
valid<- train_df[split == 2,]
test<- train_df[split == 3, ]

#pairs(train[,c(2,3,5:8,10,12)])

str(train)
md1<- glm(Survived ~ Pclass*Sex*Age+SibSp+Parch+Embarked+Deck+num_cab, data = train, family = binomial)
summary(md1)

preds<- predict(md1, newdata = valid, type = "response") >0.68
mean(preds == valid$Survived, na.rm = T)
valid[is.na(preds),]



