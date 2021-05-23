library(ggplot2)
library(tidyr)
library(dplyr)
library(stringr)

script.dir <- dirname(sys.frame(1)$ofile)
dir <- "tac09_A"
path <- sprintf("%s/%s/sents.csv", script.dir, dir)

data <- read.csv(path)

data.summ <- data %>% 
  group_by(group) %>%
  summarise(
    corr.R2.R.scores = cor(scores, R2.R / num_words), 
    corr.R2.R.lexrank = cor(lexrank, R2.R / num_words), 
    corr.R1.R.scores = cor(scores, R1.R / num_words), 
    corr.R1.R.lexrank = cor(lexrank, R1.R / num_words), 
    corr.R2.P.scores = cor(scores, R2.P), 
    corr.R2.P.lexrank = cor(lexrank, R2.P), 
    corr.R1.P.scores = cor(scores, R1.P ), 
    corr.R1.P.lexrank = cor(lexrank, R1.P ), 
    corr.pos.scores = cor(scores, 1 -position/doc_sents),
    corr.pos.lexrank = cor(lexrank, 1 - position / doc_sents),
    corr.tfisf.scores = cor(scores, tfisf),
    corr.tfisf.lexrank = cor(lexrank, tfisf),
    corr.btfisf.scores = cor(scores, btfisf),
    corr.btfisf.lexrank = cor(lexrank, btfisf),
    corr.words.scores = cor(scores, num_words),
    corr.words.lexrank = cor(lexrank, num_words),
    corr.nouns.scores = cor(scores, nouns + prpns),
    corr.nouns.lexrank = cor(lexrank, nouns + prpns),
  )

res <- data.summ %>% 
  ungroup %>% 
  select(-group) %>%
  summarise_all(.funs = c(mean="mean"))
print(res)

temp <- data %>% group_by(group) %>% 
  summarise(n = n()) %>%
  arrange(n)
