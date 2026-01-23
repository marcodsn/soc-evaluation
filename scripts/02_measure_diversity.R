library(readr)
library(dplyr)
library(koRpus)
library(koRpus.lang.en)

# 1. Load Data
data <- read_csv("data/processed/data.csv")

# 2. Define MTLD Helper Function
# koRpus requires a specific S4 object format, so we define a wrapper
get_mtld_korpus <- function(x) {
  # Skip empty or NA text to avoid errors
  if(is.na(x) || nchar(x) == 0) return(NA)

  tryCatch({
    tagged <- tokenize(x, lang="en", format="obj")
    return(MTLD(tagged)@MTLD$MTLD)
  }, error=function(e) NA)
}

# 3. Calculate Scores
# We use sapply to apply the function to every row.
data$mtld_score <- sapply(data$text, get_mtld_korpus)

# 4. Significance Check & Benchmark
# Remove NAs (failed calculations or empty rows) for the average
avg_score <- mean(data$mtld_score, na.rm = TRUE)

print(paste("Your Dataset Average MTLD:", round(avg_score, 2)))

# Reference Benchmarks from ConvoGen Paper [Source: Table V]:
# Human (DailyDialog): ~53.44
# Human (Empathetic):  ~63.47
# Synthetic (ConvoGen): ~85 - 129

if(is.nan(avg_score)) {
  print("ERROR: Could not calculate average. Check if data loaded correctly.")
} else if(avg_score > 63.47) {
  print("SUCCESS: Our dataset exceeds human lexical diversity benchmarks!")
} else {
  print("NOTE: Our dataset aligns closely with casual human speech patterns.")
}

# 6. Save
write_csv(data, "results/mtld_scores.csv")
