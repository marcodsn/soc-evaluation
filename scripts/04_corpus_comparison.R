library(readr)
library(dplyr)
library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(ggplot2)
library(tidytext)

# 1. Load Data
chat_data <- read_csv("data/processed/data.csv")

# Load the generated English corpus (reading line by line into a dataframe)
gen_lines <- readLines("data/raw/en_texts_dynamic.txt")
gen_data <- data.frame(text = gen_lines, stringsAsFactors = FALSE)

# 2. Create Quanteda Objects
# Create separate corpora and assign a source label to each
corpus_chat <- corpus(chat_data, text_field = "text")
docvars(corpus_chat, "Source") <- "Conversational"
docnames(corpus_chat) <- paste0("chat_", seq_len(ndoc(corpus_chat)))  # Ensure unique docnames

corpus_gen <- corpus(gen_data, text_field = "text")
docvars(corpus_gen, "Source") <- "English_Corpus"
docnames(corpus_gen) <- paste0("gen_", seq_len(ndoc(corpus_gen)))

# Combine into one master corpus for comparison
master_corpus <- corpus_chat + corpus_gen

# Tokenize and clean (remove punctuation, symbols, numbers)
master_tokens <- quanteda::tokens(master_corpus, remove_punct = TRUE, remove_symbols = TRUE, remove_numbers = TRUE)
master_dfm <- dfm(master_tokens)

# 3. Analysis 1: Global Vocab Comparison
# Retrieve the "Source" variable explicitly from the corpus
grouping_vector <- docvars(master_corpus, "Source")

# Group by Source to compare TTR side-by-side
dfm_grouped <- dfm_group(master_dfm, groups = grouping_vector)

total_tokens_chat <- ntoken(dfm_grouped)["Conversational"]
total_types_chat  <- ntype(dfm_grouped)["Conversational"]

total_tokens_gen <- ntoken(dfm_grouped)["English_Corpus"]
total_types_gen  <- ntype(dfm_grouped)["English_Corpus"]

print(paste("Conversational - Total Words:", total_tokens_chat, "| Unique:", total_types_chat))
print(paste("Conversational TTR:", round(total_types_chat / total_tokens_chat, 3)))

print(paste("English Corpus - Total Words:", total_tokens_gen, "| Unique:", total_types_gen))
print(paste("English Corpus TTR:", round(total_types_gen / total_tokens_gen, 3)))

# 4. Analysis 2: Rare Words (Hapax Legomena)
# We subset the DFM to calculate rare words per specific dataset
dfm_chat <- dfm_subset(master_dfm, Source == "Conversational")
freq_chat <- textstat_frequency(dfm_chat)
rare_chat <- freq_chat %>% filter(frequency == 1)

dfm_gen <- dfm_subset(master_dfm, Source == "English_Corpus")
freq_gen <- textstat_frequency(dfm_gen)
rare_gen <- freq_gen %>% filter(frequency == 1)

print(paste("Rare Words in Conversational:", nrow(rare_chat)))
print(paste("Rare Words in English Corpus:", nrow(rare_gen)))

# 5. Analysis 3: Content Word Biases
# Remove stopwords to see content
stopwords <- c(stopwords("en"), letters)
clean_dfm <- dfm_remove(master_dfm, stopwords)

group_vector <- docvars(clean_dfm, "Source")

# Get top 15 words for each source
top_words <- textstat_frequency(clean_dfm, n = 15, groups = group_vector)

print("Top Content Words by Source:")
print(head(top_words))

# 6. Visualization: Save a comparative plot
png("results/comparative_top_words.png", width=1000, height=600)
top_words %>%
  ggplot(aes(x = reorder_within(feature, frequency, group), y = frequency, fill = group)) +
  geom_col(show.legend = FALSE) +
  scale_x_reordered() +
  coord_flip() +
  facet_wrap(~ group, scales = "free") +
  labs(title = "Most Frequent Content Words: Conversational vs English Corpus", x = NULL, y = "Frequency") +
  theme_minimal()
dev.off()

# 7. Analysis 4: Keyness (Statistical Bias)
# Identify words that are statistically 'over-represented' in our chat data compared to the corpus
keyness_stat <- textstat_keyness(dfm_grouped, target = "Conversational")

print("Top Words unique to Conversational Data (vs General English):")
print(head(keyness_stat, 10))

# Save Keyness Plot
png("results/keyness_bias.png", width=800, height=600)
textplot_keyness(keyness_stat, n = 15, color = c("steelblue", "grey")) +
  labs(title = "Distinctive Vocabulary", subtitle = "Blue = Over-used in Chat | Grey = Over-used in Corpus")
dev.off()
