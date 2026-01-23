library(quanteda)
library(quanteda.textplots)
library(readr)

# 1. Load Data
data <- read_csv("data/processed/data.csv")
chat_corpus <- corpus(data, text_field = "text")

# 2. Preprocessing for Network Analysis
# We remove stopwords + "standard" conversational fillers to see the actual topics
# custom_stopwords <- c(stopwords("en"), "just", "like", "im", "thats", "dont", "can")
custom_stopwords <- c(stopwords("en"),
                      "just", "like", "im", "thats", "dont", "can", "cant",
                      "one", "also", "really", "get", "go", "know", "think",
                      "well", "see", "good", "got", "ve", "re", "ll", "ill",
                      "said", "didnt", "back", "us", "yeah", "okay", "yes", "oh",
                      "right", "sure", "maybe", "youre", "going", "want", "hes",
                      "theyre", "isnt", "ive", "didnt", "would", "could", "much")

tokens_net <- quanteda::tokens(chat_corpus, remove_punct = TRUE, remove_numbers = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(custom_stopwords)

# 3. Create Feature Co-occurrence Matrix (FCM)
# We look for words appearing within a 5-word window of each other
chat_dfm <- dfm(tokens_net)
# Focus only on the top 50 most frequent terms to keep the graph readable
top_feats <- names(topfeatures(chat_dfm, 50))
chat_fcm <- fcm(tokens_net, context = "window", window = 5, tri = FALSE)
chat_fcm_select <- fcm_select(chat_fcm, pattern = top_feats)

# 4. Plot and Save the Network
png("results/semantic_network.png", width=1000, height=1000)
textplot_network(chat_fcm_select,
                 min_freq = 0.8,  # We need to reduce clutter by showing only stronger connections
                 edge_alpha = 0.4,
                 edge_size = 2,
                 edge_color = "grey70",
                 vertex_labelsize = 5,
                 vertex_labelcolor = "black",
                 vertex_color = "steelblue")
dev.off()

print("Network Analysis Complete. Graph saved to results/semantic_network.png")
