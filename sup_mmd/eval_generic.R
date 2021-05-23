library(ggplot2)
library(tidyr)
library(dplyr)
library(stringr)

thresholds = c(0.36, 0.09)
evals = c('ROUGE-1', 'ROUGE-2')
intercepts = c(0.39, 0.1)
limits <- matrix(NA, nrow=2, ncol=2)
limits[, 1] <- c(0.37, 0.405)
limits[, 2] <- c(0.09, 0.11)
dd <- "/DUC03"

# thresholds = c(0.01, 0.01)
# evals = c('ROUGE-SU4', 'ROUGE-2')
# intercepts = c(0.15, 0.12)
# limits <- matrix(NA, nrow=2, ncol=2)
# limits[, 1] <- c(0.125, 0.155)
# limits[, 2] <- c(0.10, 0.125)
# dd <- "/TAC08A_R2r0_run10/"
# 
# thresholds = c(0.01, 0.01)
# evals = c('ROUGE-SU4', 'ROUGE-2')
# intercepts = c(0.138, 0.101)
# limits <- matrix(NA, nrow=2, ncol=2)
# limits[, 1] <- c(0.12, 0.145)
# limits[, 2] <- c(0.08, 0.11)
# dd <- "/TAC08Bg_run10/"

script.dir <- paste0(dirname(sys.frame(1)$ofile), dd)

# dataset,gamma,lr,beta,alpha,sf,kernels,r,eval,metric,score

data_test_retrained <- read.csv(sprintf("%s/gs_test.txt.csv", script.dir) ) %>%
  filter(metric == "R", cv == "x", eval %in% evals) %>%
  mutate(lambdaa = "") %>%
  select(-c("metric", "cv", "lambdaa")) %>%
  rename(test = score)

data_train <- read.csv(sprintf("%s/gs_train.txt.csv", script.dir) ) %>%
  filter(metric == "R", cv == "x", eval %in% evals) %>%
  mutate(lambdaa = "") %>%
  select(-c("metric", "cv", "lambdaa")) %>%
  rename(train = score)

### best performing model on retrained models across hyperparameters
data.retrained <- inner_join(data_test_retrained, data_train,
                   by = c("gamma", "beta", "r", "eval", "alpha", "epoch" )) %>%
  mutate(r = factor(r), beta = factor(beta), alpha = factor(alpha), gamma = factor(gamma)) #%>%
  # filter(gamma %in% c(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), beta %in% c(0.5, 1.0, 2.0, 4.0),
  #        r %in% c(0.0, 0.02, 0.04))

n1 <- sprintf("train_%s", evals[1])
n2 <- sprintf("train_%s", evals[2])

res.retrained <- data.retrained %>%
    pivot_wider(id_cols = c("gamma", "beta", "r", "alpha", "epoch"),
                names_from = "eval",
                values_from = c("train", "test" ) ) %>%
    mutate(hm_train = 2 * get(n1) * get(n2) / (get(n1) + get(n2)) ) %>%
    mutate(score = get(n2) ) %>%
    group_by(alpha, r) %>%
    filter(score == max(score)) %>%
    arrange(alpha, score) %>%
    mutate(hm_train = round(hm_train, 5), score = round(score, 5))

write.csv(res.retrained,
          sprintf("%s/res.%s.csv", script.dir, "R"),
          row.names = F, quote = F
)