# Install and load pacman package 
if(!require(pacman)) install.packages("pacman", repos = "http://cran.us.r-project.org")

# Load required libraries using pacman
pacman::p_load(tidyverse, ggplot2, ggthemes, data.table, lubridate, caret, 
               knitr, scales, treemapify)

# Download the MovieLens dataset and read the ratings file
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# Read the movies file and split it into columns
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

# Merge ratings and movies data
movielens <- left_join(ratings, movies, by = "movieId")

# Create training and test sets
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Ensure the test set only contains users and movies present in the training set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Clean up temporary variables
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Create training and test sets from the edx set
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-test_index,]
edx_temp <- edx[test_index,]

# Ensure the test set only contains users and movies present in the training set
edx_test <- edx_temp %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

removed <- anti_join(edx_temp, edx_test)
edx_train <- rbind(edx_train, removed)

# Clean up temporary variables
rm(edx_temp, test_index, removed)

# EDA for the edx dataset
edx %>% as_tibble()
glimpse(edx)

# Summarize the edx dataset
edx %>% summarize(unique_users = length(unique(userId)),
                  unique_movies = length(unique(movieId)),
                  unique_genres = length(unique(genres)))

summary(edx$rating)

# Plot the distribution of ratings
ggplot(edx, aes(x = rating)) +
  geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
  labs(x = "Rating", y = "Count", title = "Distribution of Ratings")

# Calculate the number of ratings by year
ratings_by_year <- edx %>% 
  group_by(RatingYear) %>% 
  summarise(Ratings_Count = n())

# Plot the number of ratings over time
ggplot(ratings_by_year, aes(x = RatingYear, y = Ratings_Count)) +
  geom_line(color = "blue") +
  labs(x = "Rating Year", y = "Number of Ratings", title = "Number of Ratings Over Time") +
  theme_minimal()

# Unique ratings
unique_ratings <- unique(edx$rating)
sort(unique_ratings)
edx %>% group_by(rating) %>% summarize(ratings_sum = n()) %>%
  arrange(desc(ratings_sum))

# Proportion of ratings greater than or equal to 3
rp <- edx %>% filter(edx$rating >= 3)
nrow(rp) / length(edx$rating)

# Convert timestamps to POSIXct and extract year
edx <- edx %>% mutate(timestamp = as.POSIXct(timestamp, origin = "1970-01-01", tz = "EST"))
edx$timestamp <- format(edx$timestamp, "%Y")
names(edx)[names(edx) == "timestamp"] <- "RatingYear"
head(edx)

# Repeat for validation and edx_train sets
validation <- validation %>% mutate(timestamp = as.POSIXct(timestamp, origin = "1970-01-01", tz = "EST"))
validation$timestamp <- format(validation$timestamp, "%Y")
names(validation)[names(validation) == "timestamp"] <- "RatingYear"
head(validation)

edx_train <- edx_train %>% mutate(timestamp = as.POSIXct(timestamp, origin = "1970-01-01", tz = "EST"))
edx_train$timestamp <- format(edx_train$timestamp, "%Y")
names(edx_train)[names(edx_train) == "timestamp"] <- "RatingYear"
head(edx_train)

edx_test <- edx_test %>% mutate(timestamp = as.POSIXct(timestamp, origin = "1970-01-01", tz = "EST"))
edx_test$timestamp <- format(edx_test$timestamp, "%Y")
names(edx_test)[names(edx_test) == "timestamp"] <- "RatingYear"
head(edx_test)

range(edx$RatingYear)

# Convert RatingYear to numeric and check the structure
edx$RatingYear <- as.numeric(edx$RatingYear)
str(edx)

# Summarize ratings by year and title
edx %>% group_by(RatingYear, title) %>% 
  summarize(Ratings_Sum = n(), Average_Rating = mean(rating)) %>%
  mutate(Average_Rating = sprintf("%0.2f", Average_Rating)) %>%
  arrange(-Ratings_Sum) %>% print(n = 50)

# Separate genres into individual rows
edx_genres <- edx %>% separate_rows(genres, sep = "\\|")

# Summarize ratings by genre
edx_genres %>% 
  group_by(genres) %>% summarize(Ratings_Sum = n(), Average_Rating = mean(rating)) %>%
  arrange(-Ratings_Sum)

# Summarize ratings by genre and sort by average rating
edx_genres %>%
  group_by(genres) %>% summarize(Ratings_Sum = n(), Average_Rating = mean(rating)) %>%
  arrange(-Average_Rating)

# Convert genres to factors
edx$genres <- as.factor(edx$genres)
edx_genres$genres <- as.factor(edx_genres$genres)
class(edx_genres$genres)

# Extract release year from the title
yearreleaseda <- as.numeric(str_sub(edx$title, start = -5, end = -2))
edx <- edx %>% mutate(yearReleased = yearreleaseda)
head(edx)

# Repeat for validation, edx_train, and edx_test sets
yearreleasedb <- as.numeric(str_sub(validation$title, start = -5, end = -2))
validation <- validation %>% mutate(yearReleased = yearreleasedb)
head(validation)

yearreleasedc <- as.numeric(str_sub(edx_train$title, start = -5, end = -2))
edx_train <- edx_train %>% mutate(yearReleased = yearreleasedc)
head(edx_train)

yearreleasedd <- as.numeric(str_sub(edx_test$title, start = -5, end = -2))
edx_test <- edx_test %>% mutate(yearReleased = yearreleasedd)
head(edx_test)

# Calculate the age of the movie
edx <- edx %>% mutate(MovieAge = 2020 - yearReleased)
validation <- validation %>% mutate(MovieAge = 2020 - yearReleased)
edx_train <- edx_train %>% mutate(MovieAge = 2020 - yearReleased)
edx_test <- edx_test %>% mutate(MovieAge = 2020 - yearReleased)

# Summary statistics of MovieAge
summary(edx$MovieAge)

# Function to calculate RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Baseline model using mean rating
edx_train_mu <- mean(edx_train$rating)
NRMSE_M1 <- RMSE(edx_test$rating, edx_train_mu)
results_table <- tibble(Model_Type = "NRMSE", RMSE = NRMSE_M1) %>% 
  mutate(RMSE = sprintf("%0.4f", RMSE))
results_table

# Model using median rating
edx_train_median <- median(edx_train$rating)
MM_M2 <- RMSE(edx_test$rating, edx_train_median)
results_table <- tibble(Model_Type = c("NRMSE", "Median_Model"),
                        RMSE = c(NRMSE_M1, MM_M2)) %>% 
  mutate(RMSE = sprintf("%0.4f", RMSE))
results_table

# Model with movie effects
bi <- edx_train %>% group_by(movieId) %>%
  summarize(b_i = mean(rating - edx_train_mu))

prediction_bi <- edx_train_mu + edx_test %>%
  left_join(bi, by = "movieId") %>% .$b_i
MEM_M3 <- RMSE(edx_test$rating, prediction_bi)
results_table <- tibble(Model_Type = c("NRMSE", "Median_Model", "Movie Effects"),
                        RMSE = c(NRMSE_M1, MM_M2, MEM_M3)) %>% 
  mutate(RMSE = sprintf("%0.4f", RMSE))
results_table

# Model with movie and user effects
bu <- edx_train %>% left_join(bi, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - edx_train_mu - b_i))

prediction_bu <- edx_test %>% 
  left_join(bi, by = "movieId") %>%
  left_join(bu, by = "userId") %>%
  mutate(pred = edx_train_mu + b_i + b_u) %>%
  .$pred
UEM_M4 <- RMSE(edx_test$rating, prediction_bu)
results_table <- tibble(Model_Type = c("NRMSE", "Median_Model", "Movie Effects", "Movie and User Effects"),
                        RMSE = c(NRMSE_M1, MM_M2, MEM_M3, UEM_M4)) %>% 
  mutate(RMSE = sprintf("%0.4f", RMSE))
results_table

# Regularized movie and user effects model
lambda <- 5
bi_reg <- edx_train %>% group_by(movieId) %>%
  summarize(b_i = sum(rating - edx_train_mu)/(n() + lambda), n_i = n())

bu_reg <- edx_train %>% left_join(bi_reg, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - edx_train_mu - b_i)/(n() + lambda), n_u = n())

prediction_reg <- edx_test %>%
  left_join(bi_reg, by = "movieId") %>%
  left_join(bu_reg, by = "userId") %>%
  mutate(pred = edx_train_mu + b_i + b_u) %>% .$pred
UEM_M5 <- RMSE(edx_test$rating, prediction_reg)
results_table <- tibble(Model_Type = c("NRMSE", "Median_Model", "Movie Effects", "Movie and User Effects", 
                                       "Regularized Movie and User Effects"),
                        RMSE = c(NRMSE_M1, MM_M2, MEM_M3, UEM_M4, UEM_M5)) %>% 
  mutate(RMSE = sprintf("%0.4f", RMSE))
results_table

# Display results in a table
results_table %>% kable()
results_table$RMSE <- as.numeric(results_table$RMSE)
results_table$Model_Type <- as.factor(results_table$Model_Type)
ggplot(results_table, aes(x = Model_Type, y = RMSE)) +
  geom_bar(stat = "identity", fill = "Blue") +
  ggtitle("RMSE results per Model") +
  theme_hc()
