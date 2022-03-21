## libraries
library(tidyverse)
library(here)
library(stringr)
library(xgboost)
library(parsnip)
library(tidymodels)
library(glmnet)
library(ggmosaic)
## Function

cabins<- function(Cabin){
  len<- c()
  cab<- ifelse(Cabin == "", NA, Cabin)
  for (i in 1:length(cab)){
    if (!is.na(cab[i])) {
      len<- c(len,length(str_split(cab[i], " ")[[1]]))}
    else {len<- c(len,0)}
  }
  len
}
#cabins(train_df$Cabin) %>% head()

## Data
data_col<- c('Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare','Embarked','Deck','num_cab', "cabin_num")

train_df<- read.csv(here("train.csv")) %>%
  mutate(Survived = ifelse(Survived == 1, TRUE, FALSE),
         Sex = factor(Sex),
         Embarked = ifelse(Embarked == "", "none", Embarked),
         Embarked = factor(Embarked, levels = c("C", "Q", "S")),
         Deck = str_extract(Cabin, "[A,B,C,D,E]"),
         Deck = ifelse(is.na(Deck), "none", Deck),
         Deck = factor(Deck),
         Pclass = factor(Pclass),
         last = str_extract(Name, "[a-zA-Z]+"),
         num_cab = cabins(Cabin),
         Survived = factor(Survived),
         cabin_num = ifelse(Cabin == "", 0, as.numeric(str_extract(Cabin, '[0-9]+')))
  ) 

train_df %<>%
  replace_na(list(Age=mean(train_df$Age, na.rm = T))) %>%
  select(all_of(data_col))%>%
  filter(!is.na(cabin_num)) %>%
  filter(!is.na(Embarked))

summary(train_df)

## data split
set.seed(467)
rows<- nrow(train_df)
split<- sample(seq(1, 2), size = rows, replace = TRUE, prob = c(.8, .2))

train<- train_df[split == 1,]
valid<- train_df[split == 2,]
#test<- train_df[split == 3, ]

### graphic analysis
str(train)

## survived 
ggplot(train, aes(Survived))+
  geom_bar()

## Pclass: appears to be an association between class and survival 
ggplot(train, aes(Pclass, fill= Survived))+ geom_bar(position = "dodge")

## sex: about twice as many men as women
ggplot(train, aes(Sex, fill=Survived))+geom_bar()

## Age: appear about symmetric with peak around 30
ggplot(train, aes(Age, fill = Survived))+geom_density(alpha = 0.5)

## SibSp is strongly skewed, a log(x+1) transformation is used to help
ggplot(train, aes(log(SibSp+1), fill = Survived))+geom_density(alpha = 0.5)

## Parch: added log transformation
ggplot(train, aes(log(Parch+1), fill = Survived))+geom_density(alpha = 0.5)

## Fare: Log transformation to help with skewness
ggplot(train, aes(Fare, fill=Survived))+ geom_density()
ggplot(train, aes(log(Fare+1), fill=Survived))+ geom_density()

## Embarked
ggplot(train, aes(Embarked, fill = Survived))+ geom_bar()

## Deck
ggplot(train)+geom_mosaic(aes(x= product(Deck), fill = Survived))

## number of cabins
ggplot(train)+geom_mosaic(aes(x=product(num_cab), fill = Survived))

## cabin_num
ggplot(train, aes(x = cabin_num, fill = Survived))+ geom_histogram()+ facet_wrap(~Deck, scales = "free_y")

update<- function(df){
  df <- df %>% mutate(fam = log(SibSp+Parch+1),
                      SibSp = log(SibSp+1),
                      Parch = log(Parch+1),
                      Fare = log(Fare+1)
                      )
}

train<- update(train)
valid<- update(valid)

## Preprocess data
#Pclass*Sex*Age+SibSp+Parch+Embarked+Deck+num_cab
tita_rec<- 
  recipe(Survived ~ ., data = train)%>%
  step_dummy(c(Pclass, Sex, Embarked, Deck))%>%
  step_interact(~ Age:starts_with('Sex')) %>%
  step_interact(~ starts_with("Sex"):starts_with("Pclass"))%>%
  step_interact(~ starts_with("Deck"):cabin_num)%>%
  step_interact(~ Fare:starts_with("Embarked"))%>%
  step_poly(c(Age, SibSp, Parch, Fare), degree = 3)%>%
  step_corr(all_predictors(), -all_outcomes())%>%
  step_center(all_predictors())%>%
  step_scale(all_predictors())%>%
  prep(training = train, retain=T)


train_data<- bake(tita_rec, train)
valid_data<- bake(tita_rec, valid)
#test_data<- bake(tita_rec, test)
summary(train_data)

## cross validation
train_cv<- train_data %>%
  vfold_cv(v=10)


## logistic model building

tita_md<- logistic_reg(mode = "classification",
                       penalty = tune(), 
                       mixture = tune())%>%
  set_engine("glmnet", nlambda = 150, family = "binomial")

log_Params <- dials::parameters(
  penalty(),
  mixture()
)

log_Grid<- grid_max_entropy(log_Params, size = 100)

log_wf<- workflow() %>%
  add_model(tita_md) %>%
  add_formula(Survived~.)

log_Tuned <- tune_grid(
  object = log_wf,
  resamples = train_cv,
  grid      = log_Grid,
  metrics   = metric_set(accuracy),
  control   = control_grid(verbose = TRUE))  
#log_Tuned$.metrics

log_Tuned %>% show_best(metric = "accuracy")


log_Tuned %>% collect_metrics() %>% 
  select(mean,penalty:mixture) %>% data.table::data.table()%>%
  data.table::melt(id="mean") %>% 
  ggplot(aes(y=mean,x=value,colour=variable)) + 
  geom_point(show.legend = FALSE) + 
  facet_wrap(variable~. , scales="free") + theme_bw() +
  labs(y="accuracy", x = "Parameter")

log_BestParams <- log_Tuned %>% select_best("accuracy")

log_model_final <- tita_md %>% finalize_model(log_BestParams)
log_TrainFit<-log_model_final %>% fit(Survived~., data=train_data)
log_TrainFit


log_preds<- log_TrainFit %>% predict_classprob.model_fit(new_data=valid_data)

cuts <- seq(0,1,0.01)
log_cuts<- data.frame(cut = c(0), 
                      acc = c(0)) %>% filter(cut !=0)
for (i in cuts){
 log_cuts <- bind_rows(log_cuts,  
                       c(cut = i, acc = mean((log_preds$`TRUE`> i) == valid_data$Survived)))
}

ggplot(log_cuts, aes(cuts, acc))+
  geom_line() ## 50% appears to be best cut

log_cuts[which.max(log_cuts$acc),]

## random forest

rf_md<- rand_forest(trees = tune(), 
                    min_n = tune(), 
                    mtry = tune())%>%
  set_mode("classification")%>%
  set_engine("ranger")

rf_Params<- dials::parameters(
  trees(),
  min_n(),
  finalize(mtry(),select(train_data,-Survived))
)

rf_Grid<- grid_max_entropy(rf_Params, size = 100)


rf_wf<- workflow() %>%
  add_model(rf_md) %>%
  add_formula(Survived~.)

rf_Tuned <- tune_grid(
  object = rf_wf,
  resamples = train_cv,
  grid      = rf_Grid,
  metrics   = metric_set(accuracy),
  control   = control_grid(verbose = TRUE))  
rf_Tuned$.metrics


rf_Tuned %>% show_best(metric = "accuracy")


rf_Tuned %>% collect_metrics() %>% 
  select(mean,mtry:min_n) %>% data.table::data.table()%>%
  data.table::melt(id="mean") %>% 
  ggplot(aes(y=mean,x=value,colour=variable)) + 
  geom_point(show.legend = FALSE) + 
  facet_wrap(variable~. , scales="free") + theme_bw() +
  labs(y="accuracy", x = "Parameter")

rf_BestParams <- rf_Tuned %>% select_best("accuracy")

rf_model_final <- rf_md %>% finalize_model(rf_BestParams)
rf_TrainFit<-rf_model_final %>% fit(Survived~., data=train_data)
rf_TrainFit

rf_preds<- rf_TrainFit %>% predict_classprob.model_fit(new_data=valid_data)

rf_cuts<- data.frame(cut = c(0), 
                      acc = c(0)) %>% filter(cut !=0)
for (i in cuts){
  rf_cuts <- bind_rows(rf_cuts,  
                        c(cut = i, acc = mean((rf_preds$`TRUE`>i) == valid_data$Survived)))
}

ggplot(rf_cuts, aes(cuts, acc))+
  geom_line() ## 50% appears to be best cut

rf_cuts[which.max(rf_cuts$acc),]

mean((rf_preds$`TRUE`>0.45) == valid_data$Survived)

roc_curve(rf_preds %>% mutate(truth = valid_data$Survived),truth, `TRUE`)%>% autoplot()


## xgboost

xgb_md<- boost_tree(mode = "classification",
                    trees = tune(),
                    learn_rate = tune(),
                    sample_size = tune(),
                    mtry = tune(),
                    min_n = tune(), 
                    tree_depth = tune()
                    ) %>%
  set_engine("xgboost", objective = "binary:logistic", booster = "gbtree", 
             lambda = tune(), alpha = tune(), num_class = 2, verbose=1) 

xgboostParams <- dials::parameters(
  min_n(),
  tree_depth(),
  learn_rate(),
  finalize(mtry(),select(train_data,-Survived)),
  sample_size = sample_prop(c(0.4, 0.9)), 
  alpha = penalty_L1(),
  lambda = penalty_L2(),
  trees()
)

xgGrid<- grid_max_entropy(xgboostParams, size = 500)

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
#xgTuned$.metrics


xgTuned %>% show_best(metric = "accuracy")


xgTuned %>% collect_metrics() %>% 
  select(mean,mtry:alpha) %>% data.table::data.table()%>%
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
mean((preds$`TRUE`>0.45) == valid_data$Survived)



## neural network
## parsnip not currently working with KERAS due to keras update
# 
# nn_md <- mlp(mode = "classification",
#              hidden_units = tune(),
#              dropout = tune(),
#              #penalty = tune(),
#              epochs = 10) %>%
#   set_mode("classification") %>%
#   set_engine("keras", verbose = 0, version = 2.6)
# nn_md
# 
# nn_params<- dials::parameters(#penalty(),
#                               hidden_units(),
#                               dropout())
# 
# nngrid<- grid_max_entropy(nn_params, size=3)
# 
# nn_wf<- workflow() %>%
#   add_model(nn_md) %>%
#   add_formula(Survived~.)
# 
# nnTuned <- tune_grid(
#   object = nn_wf,
#   resamples = train_cv,
#   grid      = nngrid,
#   metrics   = metric_set(accuracy),
#   control   = control_grid(verbose = FALSE))
# nnTuned$.notes[[1]]$.notes


## Support Vector Machine

svm_md <- svm_rbf(cost = tune(), rbf_sigma = tune())%>%
  set_engine("kernlab")%>%
  set_mode("classification")

svm_params <- parameters(cost(),
                         rbf_sigma())

svm_grid<- grid_max_entropy(svm_params, size = 100)

svm_wf <- workflow() %>%
  add_model(svm_md) %>% 
  add_formula(Survived~.)

svmTuned<- tune_grid(
  object = svm_wf,
  resamples = train_cv,
  grid = svm_grid,
  metrics = metric_set(accuracy),
  control = control_grid(verbose = T)
  )

svmTuned %>% show_best(metric = "accuracy")

svmTuned %>% collect_metrics() %>%
  select(mean, cost:rbf_sigma) %>% data.table::data.table()%>%
  data.table::melt(id = "mean") %>%
  ggplot(aes(y=mean,x=value,colour=variable)) + 
  geom_point(show.legend = FALSE) + 
  facet_wrap(variable~. , scales="free") + theme_bw() +
  labs(y="accuracy", x = "Parameter")


svmBestParams <- svmTuned %>% select_best("accuracy")

svm_model_final <- svm_md %>% finalize_model(svmBestParams)
svmTrainFit<-svm_model_final %>% fit(Survived~., data=train_data)
svmTrainFit

preds<- svmTrainFit %>% predict_classprob.model_fit(new_data=valid_data)
mean((preds$`TRUE`>0.45) == valid_data$Survived)



## nnet
nn_md <- mlp(hidden_units = tune,
             penalty = tune())+
  set_engine("nnet")+
  set_mode("classification")
