library(ggplot2)
library(tidyr)
library(dplyr)

script.dir <- paste0(dirname(sys.frame(1)$ofile), "/")
data <- read.csv(paste0(script.dir, "vocab-stem.csv" ))

data <- data %>%
  group_by(term) %>%
  gather("measure", "value", c("df","tf"))

gg <- ggplot(data, aes(x = value, color = measure, fill = measure)) +
  geom_histogram(binwidth=1, position = "identity", alpha = 0.33) + 
  scale_x_continuous(breaks=c(2,4,8,16,32,64,128,256,512,1024,2056), trans = "log2", expand=c(0,0)) +
  theme_bw() + 
  facet_grid(~measure, scales = "free") +
  labs(x = "",  y = "freq")

ggsave(paste0(script.dir, "vocab-stem.pdf"), plot = gg,
       width = 12 , height = 4 , units = "in")