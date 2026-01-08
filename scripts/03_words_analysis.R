library(readr)
library(dplyr)
library(quanteda)
library(quanteda.textstats)
library(ggplot2)

# 1. Load Data
data <- read_csv("data/processed/data.csv")

# 2. Create Quanteda Object (Better for frequency stats than koRpus)
# We treat the whole column as one corpus
chat_corpus <- corpus(data, text_field = "text")
# Specify the use of quanteda, just in case koRpus is still loaded in the session (it may cause conflict errors otherwise)
chat_tokens <- quanteda::tokens(chat_corpus, remove_punct = TRUE, remove_symbols = TRUE, remove_numbers = TRUE)
chat_dfm <- dfm(chat_tokens)

# 3. Analysis 1: Global Vocab
# Calculate Total Unique Words (Types) vs Total Words (Tokens)
total_tokens <- sum(ntoken(chat_dfm))
total_types  <- ntype(chat_dfm)

print(paste("Total Words Spoken:", total_tokens))
print(paste("Total Unique Words:", total_types))
print(paste("Global TTR (Unique/Total):", round(total_types / total_tokens, 3)))

# 4. Analysis 2: Rare Words (Hapax Legomena)
# Get frequency stats
freq_stats <- textstat_frequency(chat_dfm)

# Filter for words used exactly ONCE
rare_words <- freq_stats %>% filter(frequency == 1)

print(paste("Number of Rare Words (used only once):", nrow(rare_words)))
print("Examples of Rare Words:")
print(head(rare_words$feature, 15))

# 5. Analysis 3: AI Words Biases (Most Frequent Non-Stopwords)
# We remove standard English stopwords (the, and, is) to see the "content"
clean_dfm <- dfm_remove(chat_dfm, stopwords("en"))
top_words <- textstat_frequency(clean_dfm, n = 20)

print("Top 20 Most Used Content Words (The AI's 'Favorite' words):")
print(top_words[, c("feature", "frequency")])

# 6. Visualization: Save a plot of top words
png("results/top_words.png", width=800, height=600)
data.frame(top_words) %>%
  ggplot(aes(x = reorder(feature, frequency), y = frequency)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Most Frequent Words in AI Chat", x = NULL, y = "Frequency") +
  theme_minimal()
dev.off()
