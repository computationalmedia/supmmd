library(ggplot2)
library(tidyr)
library(dplyr)
library(stringr)

thresholds = c(0.11, 0.08)
evals = c('ROUGE-SU4', 'ROUGE-2')
intercepts = c(0.13, 0.096)
limits <- matrix(NA, nrow=2, ncol=2)
limits[, 1] <- c(0.12, 0.145)
limits[, 2] <- c( 0.090, 0.104 )
dd <- "/TAC0AB_lin1/"

script.dir <- paste0(dirname(sys.frame(1)$ofile), dd)

# dataset,gamma,lr,beta,alpha,sf,kernels,r,eval,metric,score

data_test_retrained <- read.csv(sprintf("%s/gs_test.txt.csv", script.dir) ) %>%
  filter(metric == "R", cv == "x", eval %in% evals) %>%
  select(-c("metric", "cv")) %>%
  rename(test = score)

data_train <- read.csv(sprintf("%s/gs_train.txt.csv", script.dir) ) %>%
  filter(metric == "R", cv == "x", eval %in% evals) %>%
  select(-c( "metric", "cv")) %>%
  rename(train = score)

### best performing model on retrained models across hyperparameters
data.retrained <- inner_join(data_test_retrained, data_train,
                             by = c("gamma", "beta", "r", "eval", "alpha", "epoch", "lambdaa" )) %>%
  # filter(gamma < 1.0) %>%
  mutate(r = factor(r), beta = factor(beta), lambdaa = factor(lambdaa),
         alpha = factor(alpha), gamma = factor(gamma)) #%>%
  # filter(beta != 0.025)

n1 <- sprintf("train_%s", evals[1])
n2 <- sprintf("train_%s", evals[2])

res.retrained <- data.retrained %>%
  pivot_wider(id_cols = c("gamma", "beta", "r", "alpha", "epoch", "lambdaa"),
              names_from = "eval",
              values_from = c("train", "test" ) ) %>%
  mutate(hm_train = 2 * get(n1) * get(n2) / (get(n1) + get(n2)) ) %>%
  mutate(score = get(n2) ) %>%
  group_by(alpha, r, lambdaa) %>%
  filter(score == max(score)) %>%
  arrange(alpha, score) %>%
  mutate(hm_train = round(hm_train, 5), score = round(score, 5))

write.csv(res.retrained,
          sprintf("%s/res.%s.csv", script.dir, "R"),
          row.names = F, quote = F
)