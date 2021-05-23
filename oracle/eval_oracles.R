library(ggplot2)
library(tidyr)
library(dplyr)
library(GGally)
library(ggpubr)
library(stringr)

script.dir <- paste0(dirname(sys.frame(1)$ofile), "/")

datasets <- c("duc03", "duc04", "tac08_icsi2", "tac09_icsi2")
# datasets <- c("tac08_icsi-expanded", "tac09_icsi-expanded")
# datasets <- c("duc03-expanded", "duc04-expanded")
leg_pos <- function(sets){
  if(sets == 1){
    return("none")
  }
  else{
    return("bottom")
  }
}

for (dataset in datasets){
  data <- read.csv(paste0(script.dir, sprintf("/%s-oracles.csv", dataset )))
  data$r <- factor(data$r)
  data$set[is.na(data$set)] <- factor('A')
  sets <- length(unique(data$set))
  temp <- data %>% 
    group_by(set, method, scorer, r) %>% 
    summarize(rouge1_P = mean(rouge1_P), rouge1_R = mean(rouge1_R), 
              rouge2_P = mean(rouge2_P), rouge2_R = mean(rouge2_R),
              cost = mean(cost), len = mean(len)) %>%
    gather(key = "eval", value = "score", rouge1_P, rouge1_R, rouge2_P, rouge2_R)
  
  data.long = data %>%
    gather(key = "eval", value = "score", rouge1_P, rouge1_R, rouge2_P, rouge2_R)
  
  gg.bar <- ggplot(temp, aes(x = scorer, y = score, fill = set)) + 
    geom_bar(stat = "identity", position = position_dodge()) +
    facet_grid(eval ~ r, scales = "free_y") +
    scale_color_manual(values=c("#999999", "#E69F00", "#56B4E9")) +
    geom_text(data=temp,aes(x=scorer, y=score * 0.8,
                            label = str_replace(as.character(
                              sprintf("%.3g", round(score, 3)) ), "^0\\.", "."), 
                            fill = set),
              inherit.aes=FALSE,angle=45, size=4, position = position_dodge(0.9)) +
    geom_text(data=temp,aes(x=scorer, y=score * 0.5,
                            label = str_replace(as.character(
                              sprintf("#:%.3g", round(cost, 3)) ), "^0\\.", "."), 
                            fill = set),
              inherit.aes=FALSE,angle=45, size=4, position = position_dodge(0.9)) +
    geom_text(data=temp,aes(x=scorer, y=score * 0.2,
                            label = str_replace(as.character(
                              sprintf("s:%.3g", round(len, 3)) ), "^0\\.", "."), 
                            fill = set),
              inherit.aes=FALSE,angle=45, size=4, position = position_dodge(0.9)) +
    labs(x = "", y = "") +
    theme_bw() + 
    theme(legend.position = leg_pos(sets))
    
  
  ggsave(paste(script.dir, "-barplot.pdf", sep = dataset), plot = gg.bar, 
         width = 12 + (sets - 1) * 8, height = 8 + (sets - 1) * 2, units = "in")
  
  # gg.box <- ggplot(data.long, aes(x = scorer, y = score, fill = set)) +
  #   geom_boxplot(position = position_dodge()) +
  #   # stat_summary(fun.y=mean, colour=set, geom="point",
  #   #              shape=18, size=3,show_guide = FALSE) +
  #   # geom_text(data=temp,aes(x=scorer, y=score*0.8,
  #   #                         label = str_replace(as.character(
  #   #                           sprintf("%.3g", round(score, 3)) ), "^0\\.", "."),
  #   #                         fill = set),
  #   #           inherit.aes=FALSE,angle=45, size=4, position = position_dodge(0.9)) +
  #   facet_grid(eval ~  r, scales = "free_y") +
  #   labs(x = "", y = "") +
  #   theme_bw() +
  #   theme(legend.position = leg_pos(sets))
  # 
  # ggsave(paste(script.dir, "-boxplot.pdf", sep = dataset), plot = gg.box,
  #        width = 12 + (sets - 1) * 7, height = 8 + (sets - 1) * 2 , units = "in")
  
  long.df <- data %>% 
    select(-idxs) %>% 
    gather("metric","score", c("rouge1_P", "rouge2_P", "rouge1_R", "rouge2_R"))
  
  gg.scatter <- ggplot(long.df %>% 
                         filter(r %in% c(0, 0.1, 0.3, 0.4, 0.5)), 
                       aes(x = len, y = score, color = r)) +
    geom_point(alpha = 0.33) +
    geom_smooth(method='lm', se = F) +
    scale_color_brewer(palette="Set1")+
    facet_grid(metric ~ scorer + set, scales = "free") +
    labs(x = "#sents") +
    theme_bw() +
    theme(legend.position = "bottom")

  ggsave(paste(script.dir, "-scatter.pdf", sep = dataset), plot = gg.scatter,
         width = 9 + (sets -1) * 6 , height = 8 , units = "in")
}
