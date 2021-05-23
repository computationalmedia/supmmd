library(ggplot2)
library(tidyr)
library(dplyr)
library(GGally)
library(ggpubr)
library(stringr)
# library(ggExtra)

script.dir <- paste0(dirname(sys.frame(1)$ofile), "/")

datasets <- c("duc03", "duc04", "tac08", "tac09")
# datasets <- c("tac08_icsi2", "tac09_icsi2")

df <- list()
method <- "auto"

agg <- function(data){
  temp <- data %>%
    group_by(dataset, set) %>%
    mutate(nouns_all = nouns + prpns) %>%
    summarise( sents = n(),
               groups = length(unique(group)),
               summ_sents = round(sum(label) / groups, 2),
               docs = length(unique(doc_id)),
               sd.num_words = round(sd(num_words), 2),
               num_words = round(mean(num_words), 2),
               doc_sents = round(sents /docs, 2),
               non_stop.sd = round(sd(vocab_words / num_words), 3),
               non_stop = round(mean(vocab_words / num_words), 3),
               tf_avg.sd = round(sd(tf_avg, na.rm = T), 3),
               tf_avg = round(mean(tf_avg, na.rm = T), 3),
               logtf_sum.sd = round(sd(logtf_sum), 3),
               logtf_sum = round(mean(logtf_sum), 3),
               isf_avg.sd = round(sd(isf_avg, na.rm = T), 3),
               isf_avg = round(mean(isf_avg, na.rm = T), 3),
               tfisf_sum.sd = round(sd(tfisf_sum), 3),
               tfisf_sum = round(mean(tfisf_sum), 3),
               logtfisf_sum.sd = round(sd(logtfisf_sum), 3),
               logtfisf_sum = round(mean(logtfisf_sum), 3),
               isf_all_avg.sd = round(sd(isf_all_avg, na.rm = T), 3),
               isf_all_avg = round(mean(isf_all_avg, na.rm = T), 3),
               tfisf_sum_all.sd = round(sd(tfisf_sum_all), 3),
               tfisf_sum_all = round(mean(tfisf_sum_all), 3),
               logtfisf_sum_all.sd = round(sd(logtfisf_sum_all), 3),
               logtfisf_sum_all = round(mean(logtfisf_sum_all), 3),
               prp.sd = round(sd(pronouns), 3),
               prp = round(mean(pronouns), 3),
               mx.prp = round(max(pronouns), 3),
               sim.sd = round(sd(sim), 3),
               sim = round(mean(sim), 3),
               sim_all.sd = round(sd(sim_all), 3),
               sim_all = round(mean(sim_all), 3),
               sd.nouns = round(sd(nouns), 3),
               nouns = round(mean(nouns), 3),
               sd.prpns = round(sd(prpns), 3),
               prpns = round(mean(prpns), 3),
               sd.nouns_all = round(sd(nouns_all), 3),
               nouns_all = round(mean(nouns_all), 3),
    )  
  return(temp)
}

i = 1
for (dataset in datasets){
  sents <- read.csv(paste0(script.dir, sprintf("/%s-sents-punkt.csv", dataset ))) %>%
    mutate(label = `y_hm_0.4`) %>%
    select(group, set, doc_id, sent_id, label, R1.P, R1.R, R2.P, R2.R)
  surface_feats <- read.csv(paste0(script.dir, sprintf("/%s-surf-feats.csv", dataset )))
  data <- inner_join(sents, surface_feats, by = c("group", "set", "sent_id", "doc_id"))
  data <- data %>%
    mutate(rel_pos = position / doc_sents)
  print(sprintf("%d,%d,%d",nrow(sents), nrow(surface_feats), nrow(data)))
  data$dataset <- dataset
  data$set[is.na(data$set)] <- "-"
  sets <- length(unique(data$set))
  if (is.null(data$title_sim)) {
    data$title_sim <- 0.0
  }
  if (is.null(data$query_sim)) {
    data$query_sim <- 0.0
  }
    if (is.null(data$narr_sim)) {
    data$narr_sim <- 0.0
  }
  
  df[[i]] <- data
  i = i + 1
  rm(sents, surface_feats)
}

data <- do.call(rbind, df)
data$set <- as.factor(data$set)
data$dataset <- factor(data$dataset )
rm(df)

print(nrow(data))
temp <- agg(data)

write.csv(temp, paste0(script.dir, "/stats-all-punkt.csv"), 
          row.names = F, quote = F)

data <- data %>%
  filter(num_words <= 55 & num_words >= 8) %>%
  mutate(data.group = substr(dataset, start = 1, stop = 3))
print(nrow(data))
temp <- agg(data)
 
write.csv(temp, paste0(script.dir, "/stats-filtered-punkt.csv"), 
          row.names = F, quote = F)

data.long <- data %>%
  gather("metric", "score", c("R1.P", "R1.R", "R2.P", "R2.R"))
data.long$metric <- factor(data.long$metric )
# 
gg.scatter <- ggplot(data.long, aes(x = rel_pos, y = score)) +
  geom_smooth(method = method, se = T) +
  # stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  # scale_fill_distiller(palette="Spectral", direction=1) +
  facet_grid(metric ~ data.group + set, scales = "free")

ggsave(paste0(script.dir, "/pos-score-punkt.pdf"), plot = gg.scatter,
       width = 16, height = 16 , units = "in")

gg.scatter3 <- ggplot(data.long, aes(x = rel_pos/num_words, y = score)) +
  # geom_point(alpha = 0.01) +
  # stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  # scale_fill_distiller(palette="Spectral", direction=1) +
  geom_smooth(method = method, se = T) +
  facet_grid(metric ~ data.group + set, scales = "free")

ggsave(paste0(script.dir, "/pos_words-score-punkt.pdf"), plot = gg.scatter3,
       width = 16, height = 16 , units = "in")

gg.scatter2 <- ggplot(data, aes(x = rel_pos, y = num_words)) +
  # geom_point(alpha = 0.01) +
  # stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  # scale_fill_distiller(palette="Spectral", direction=1) +
  geom_smooth(method = method, se = T) +
  facet_grid( ~ data.group + set, scales = "free")

# p <- ggMarginal(gg.scatter2, type="boxplot")

ggsave(paste0(script.dir, "/pos-words-punkt.pdf"), plot = gg.scatter2,
       width = 12, height = 5 , units = "in")


gg.scatter4 <- ggplot(data.long, aes(x = num_words, y = score)) +
  # geom_point(alpha = 0.01) +
  # stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  # scale_fill_distiller(palette="Spectral", direction=1) +
  geom_smooth(method = method, se = T) +
  facet_grid(metric ~ data.group + set, scales = "free")

ggsave(paste0(script.dir, "/words-score-punkt.pdf"), plot = gg.scatter4,
       width = 16, height = 16 , units = "in")

gg.scatter6 <- ggplot(data.long, aes(x = query_sim, y = score)) +
  # geom_point(alpha = 0.01) +
  # stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  # scale_fill_distiller(palette="Spectral", direction=1) +
  geom_smooth(method = method, se = T) +
  facet_grid(metric ~ data.group + set, scales = "free")

ggsave(paste0(script.dir, "/query-score-punkt.pdf"), plot = gg.scatter6,
       width = 16, height = 16 , units = "in")

gg.boxplot <- ggplot(data.long, aes(x = group, y = query_sim, fill = set)) +
  geom_boxplot() +
  facet_wrap(  dataset ~ . , scales = "free", ncol = 1) +
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggsave(paste0(script.dir, "/query_sim-punkt.pdf"), plot = gg.boxplot,
       width = 12, height = 9 , units = "in")