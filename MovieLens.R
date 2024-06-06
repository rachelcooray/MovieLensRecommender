# Load necessary packages if not already installed
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")

# Load required libraries
library(tidyverse)
library(caret)
library(glmnet)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Define file paths for ratings and movies and unzip the dataset if necessary
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

# Load ratings data
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Load movies data
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Merge ratings and movies data
movielens <- left_join(ratings, movies, by = "movieId")

# Set seed for reproducibility
set.seed(1, sample.kind="Rounding") 

# Create a train-test split
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Create a final holdout test set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Update training set after creating the final holdout test set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Clean up unnecessary objects from the environment
rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Exploratory Data Analysis (EDA)

# Summary statistics
summary(edx)
summary(final_holdout_test)

# Distribution of ratings
ggplot(edx, aes(x = rating)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Distribution of Ratings in Training Set",
       x = "Rating",
       y = "Count")

ggplot(final_holdout_test, aes(x = rating)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Distribution of Ratings in Test Set",
       x = "Rating",
       y = "Count")

# Number of ratings per user
ratings_per_user <- edx %>%
  group_by(userId) %>%
  summarise(num_ratings = n())

ggplot(ratings_per_user, aes(x = num_ratings)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  labs(title = "Number of Ratings per User in Training Set",
       x = "Number of Ratings",
       y = "Count")

# Number of ratings per movie
ratings_per_movie <- edx %>%
  group_by(movieId) %>%
  summarise(num_ratings = n())

ggplot(ratings_per_movie, aes(x = num_ratings)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  labs(title = "Number of Ratings per Movie in Training Set",
       x = "Number of Ratings",
       y = "Count")

# Convert timestamp to Date format
edx <- edx %>%
  mutate(timestamp = as.POSIXct(timestamp, origin = "1970-01-01"))

# Number of ratings over time
ratings_over_time <- edx %>%
  count(timestamp) %>%
  ggplot(aes(x = timestamp, y = n)) +
  geom_line(color = "skyblue") +
  labs(title = "Number of Ratings Over Time",
       x = "Timestamp",
       y = "Number of Ratings")

ratings_over_time


# Function to calculate NRMSE
nrmse <- function(predicted, actual) {
  rmse <- sqrt(mean((actual - predicted)^2))
  (rmse - min(actual)) / (max(actual) - min(actual))
}

# Model Building and Evaluation

# Baseline Model - Mean and Median
mean_rating <- mean(edx$rating)
median_rating <- median(edx$rating)
mean_rating
median_rating

# Baseline model using mean rating
baseline_mean <- rep(mean_rating, nrow(final_holdout_test))

# Baseline model using median rating
baseline_median <- rep(median_rating, nrow(final_holdout_test))

# Calculate RMSE and NRMSE for baseline models
rmse_baseline_mean <- sqrt(mean((final_holdout_test$rating - baseline_mean)^2))
nrmse_baseline_mean <- nrmse(baseline_mean, final_holdout_test$rating)

rmse_baseline_median <- sqrt(mean((final_holdout_test$rating - baseline_median)^2))
nrmse_baseline_median <- nrmse(baseline_median, final_holdout_test$rating)

cat("Baseline Model - Mean:\n")
cat("RMSE:", rmse_baseline_mean, "\n")
cat("NRMSE:", nrmse_baseline_mean, "\n\n")

cat("Baseline Model - Median:\n")
cat("RMSE:", rmse_baseline_median, "\n")
cat("NRMSE:", nrmse_baseline_median, "\n\n")

##############

# Movie Effects Model - Incorporating movie-specific biases (b_i)
movie_mean_rating <- edx %>%
  group_by(movieId) %>%
  summarise(mean_rating = mean(rating))

overall_mean_rating <- mean(edx$rating)

movie_effects_model <- edx %>%
  left_join(movie_mean_rating, by = "movieId") %>%
  group_by(movieId) %>%
  summarise(b_i = mean(mean_rating - overall_mean_rating))

final_holdout_test_movie_effects <- final_holdout_test %>%
  left_join(movie_effects_model, by = "movieId") %>%
  mutate(b_i = ifelse(is.na(b_i), 0, b_i)) %>%
  select(userId, movieId, rating, b_i)

rmse_movie_effects <- sqrt(mean((final_holdout_test_movie_effects$rating - (mean(edx$rating) + final_holdout_test_movie_effects$b_i))^2))
nrmse_movie_effects <- nrmse((mean(edx$rating) + final_holdout_test_movie_effects$b_i), final_holdout_test$rating)

cat("Movie Effects Model:\n")
cat("RMSE:", rmse_movie_effects, "\n")
cat("NRMSE:", nrmse_movie_effects, "\n\n")

# Calculate overall mean rating
overall_mean_rating <- mean(edx$rating)

# User Effects Model - Incorporating user-specific biases (b_u)
# Calculate user-specific biases (b_u)
user_mean_rating <- edx %>%
  group_by(userId) %>%
  summarise(mean_rating = mean(rating))

user_effects_model <- edx %>%
  left_join(user_mean_rating, by = "userId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(mean_rating - overall_mean_rating))

# Apply user-specific biases to the final holdout test set
final_holdout_test_user_effects <- final_holdout_test %>%
  left_join(user_effects_model, by = "userId") %>%
  mutate(b_u = ifelse(is.na(b_u), 0, b_u)) %>%
  select(userId, movieId, rating, b_u)

# Calculate RMSE and NRMSE for user effects model
rmse_user_effects <- sqrt(mean((final_holdout_test_user_effects$rating - (overall_mean_rating + final_holdout_test_user_effects$b_u))^2))
nrmse_user_effects <- rmse_user_effects / (max(final_holdout_test$rating) - min(final_holdout_test$rating))

cat("User Effects Model:\n")
cat("RMSE:", rmse_user_effects, "\n")
cat("NRMSE:", nrmse_user_effects, "\n\n")

# Movie Age Effects Model - Incorporating movie age effects (b_a)
edx <- edx %>%
  mutate(timestamp = as.Date(timestamp, origin = "1970-01-01"))

movie_age_effects_model <- edx %>%
  group_by(movieId) %>%
  summarise(b_a = as.numeric(difftime(mean(timestamp), as.Date("1970-01-01"), units = "days")))

final_holdout_test_movie_age_effects <- final_holdout_test %>%
  left_join(movie_age_effects_model, by = "movieId") %>%
  mutate(b_a = ifelse(is.na(b_a), 0, b_a)) %>%
  select(userId, movieId, rating, b_a)

rmse_movie_age_effects <- sqrt(mean((final_holdout_test_movie_age_effects$rating - (mean(edx$rating) + final_holdout_test_movie_age_effects$b_a))^2))
nrmse_movie_age_effects <- nrmse((mean(edx$rating) + final_holdout_test_movie_age_effects$b_a), final_holdout_test$rating)

cat("Movie Age Effects Model:\n")
cat("RMSE:", rmse_movie_age_effects, "\n")
cat("NRMSE:", nrmse_movie_age_effects, "\n\n")

# Regularize movie and user effects

# Calculate overall mean rating
overall_mean_rating <- mean(edx$rating)

# Calculate movie-specific biases (b_i)
movie_effects_model <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating) - overall_mean_rating)

# Calculate user-specific biases (b_u)
user_effects_model <- edx %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating) - overall_mean_rating)

# Combine movie and user effects in the training set
edx_with_effects <- edx %>%
  left_join(movie_effects_model, by = "movieId") %>%
  left_join(user_effects_model, by = "userId") %>%
  mutate(predicted_rating = overall_mean_rating + b_i + b_u)

# Handle missing values
edx_with_effects <- edx_with_effects %>%
  mutate(b_i = ifelse(is.na(b_i), 0, b_i),
         b_u = ifelse(is.na(b_u), 0, b_u),
         predicted_rating = ifelse(is.na(predicted_rating), overall_mean_rating, predicted_rating))

# Apply movie and user effects to the final holdout test set
final_holdout_test_with_effects <- final_holdout_test %>%
  left_join(movie_effects_model, by = "movieId") %>%
  left_join(user_effects_model, by = "userId") %>%
  mutate(b_i = ifelse(is.na(b_i), 0, b_i),
         b_u = ifelse(is.na(b_u), 0, b_u),
         predicted_rating = overall_mean_rating + b_i + b_u)

# Calculate RMSE and NRMSE for movie and user effects model
rmse_movie_user_effects <- sqrt(mean((final_holdout_test_with_effects$rating - final_holdout_test_with_effects$predicted_rating)^2))
nrmse_movie_user_effects <- rmse_movie_user_effects / (max(final_holdout_test$rating) - min(final_holdout_test$rating))

cat("Movie and User Effects Model:\n")
cat("RMSE:", rmse_movie_user_effects, "\n")
cat("NRMSE:", nrmse_movie_user_effects, "\n\n")

