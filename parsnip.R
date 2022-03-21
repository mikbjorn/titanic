## libraries
library(tidyverse)
library(here)
library(stringr)
library(xgboost)
library(parsnip)
library(tidymodels)
library(glmnet)
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
         Sex = factor(Sex),
         Embarked = ifelse(Embarked == "", NA, Embarked),
         Embarked = factor(Embarked, levels = c("C", "Q", "S")),
         Deck = str_extract(Cabin, "[A,B,C,D,E]"),
         Deck = ifelse(is.na(Deck), "none", Deck),
         Deck = factor(Deck),
         Pclass = factor(Pclass),
         last = str_extract(Name, "[a-zA-Z]+"),
         num_cab = cabins(Cabin),
         Survived = factor(Survived)
  ) 

train_df %<>%
  replace_na(list(Age=mean(train_df$Age, na.rm = T))) %>%
  select(all_of(data_col))%>%
  filter(!is.na(Embarked))

summary(train_df)

## data split
set.seed(467)
rows<- nrow(train_df)
split<- sample(seq(1, 3), size = rows, replace = TRUE, prob = c(.8, .2, .2))

train<- train_df[split == 1,]
valid<- train_df[split == 2,]
test<- train_df[split == 3, ]

## Preprocess data
#Pclass*Sex*Age+SibSp+Parch+Embarked+Deck+num_cab
tita_rec<- 
  recipe(Survived ~ ., data = train)%>%
  step_dummy(c(Pclass, Sex, Embarked, Deck))%>%
  #step_interact(~ Age:starts_with('Sex')) %>%
  step_interact(~ starts_with('Pclass')*Age*starts_with('Sex')) %>%
  step_corr(all_predictors(), -all_outcomes())%>%
  step_center(all_predictors())%>%
  step_scale(all_predictors())%>%
  prep(training = train, retain=T)

train_data<- bake(tita_rec, train)
valid_data<- bake(tita_rec, valid)
test_data<- bake(tita_rec, test)
summary(train_data)

## logistic model building

tita_md<- logistic_reg(penalty = 0.1)

lm_tita_md<- tita_md %>%
  set_engine("glmnet")


lm_fit<- lm_tita_md %>% 
  fit(Survived~., data = train_data)
lm_fit

preds<- predict(lm_fit, new_data = valid_data, type = "prob")
mean((preds[,2]>0.50) == valid$Survived, na.rm = T)
#valid[is.na(preds),]

## random forest

rf_md<- rand_forest(trees = 300, min_n = 4, mtry = 11)%>%
  set_mode("classification")%>%
  set_engine("ranger")

#str(train_data)

rf_md_fit<- rf_md %>% fit(Survived ~., data= train_data)

rf_preds<- predict(rf_md_fit, new_data = valid_data)
mean(rf_preds$.pred_class == valid_data$Survived)


## xgboost

train_cv<- train_data %>%
  vfold_cv(v=10)

xgb_md<- boost_tree(mode = "classification",
                    trees = 300, 
                    learn_rate = tune(),
                    sample_size = tune(),
                    mtry = tune(),
                    min_n = tune(), 
                    tree_depth = tune()
                    ) %>%
  set_engine("xgboost", objective = "binary:logistic", 
             lambda = 0, alpha = 1, num_class = 3, verbose=1) 

xgboostParams <- dials::parameters(
  min_n(),
  tree_depth(),
  learn_rate(),
  finalize(mtry(),select(train_data,-Survived)),
  sample_size = sample_prop(c(0.4, 0.9))
)

xgGrid<- grid_max_entropy(xgboostParams, size = 100)

#xgb_md_fit<- xgb_md %>% fit(Survived~., data = train_data, eval_metric = "error")

xgb_wf<- workflow() %>%
  add_model(xgb_md) %>%
  add_formula(Survived~.)

xgTuned <- tune_grid(
  object = xgb_wf,
  resamples = train_cv,
  grid      = xgGrid,
  metrics   = metric_set(accuracy),
  control   = control_grid(verbose = TRUE))  
xgTuned$.metrics


xgTuned %>% show_best(metric = "accuracy")


xgTuned %>% collect_metrics() %>% 
  select(mean,mtry:sample_size) %>% data.table::data.table()%>%
  data.table::melt(id="mean") %>% 
  ggplot(aes(y=mean,x=value,colour=variable)) + 
  geom_point(show.legend = FALSE) + 
  facet_wrap(variable~. , scales="free") + theme_bw() +
  labs(y="accuracy", x = "Parameter")

xgBestParams <- xgTuned %>% select_best("accuracy")

xgboost_model_final <- xgb_md %>% finalize_model(xgBestParams)
xgTrainFit<-xgboost_model_final %>% fit(Survived~., data=train_data)
xgTrainFit

preds<- xgTrainFit %>% predict_classprob.model_fit(new_data=valid_data)
mean((preds$`TRUE`>0.6) == valid_data$Survived)
