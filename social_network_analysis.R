# ===============================================================================
# SOCIAL NETWORK ADS ANALYSIS
# Method: Logistic Regression Classification
# ===============================================================================

# 1. LOAD LIBRARIES
# ===============================================================================
# Set CRAN mirror for package installation
options(repos = c(CRAN = "https://cloud.r-project.org"))

required_packages <- c("ggplot2", "dplyr", "caTools", "ROCR", "caret")
for(package in required_packages) {
  if(!require(package, character.only = TRUE)) {
    tryCatch({
      install.packages(package, dependencies = TRUE)
      library(package, character.only = TRUE)
    }, error = function(e) {
      cat("Failed to install", package, "\n")
    })
  }
}

# 2. DATA LOADING AND PREPROCESSING
# ===============================================================================
# Load dataset
dataset <- read.csv("Social_Network_Ads.csv")

# Basic inspection
cat("=== DATASET STRUCTURE ===\n")
str(dataset)

# Encode categorical variables
dataset$Gender <- factor(dataset$Gender, 
                         levels = c('Male', 'Female'),
                         labels = c(1, 2))
dataset$Purchased <- factor(dataset$Purchased, 
                            levels = c(0, 1))

# Drop User ID (not useful for prediction)
dataset <- dataset %>% select(-User.ID)

# Split the dataset into Training and Test set
set.seed(123)
split <- sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
# Scale only numerical columns (Age and EstimatedSalary) which are at indices 2 and 3
training_set[, 2:3] <- scale(training_set[, 2:3])
test_set[, 2:3] <- scale(test_set[, 2:3])

cat("\nData split completed.\n")
cat("Training set size:", nrow(training_set), "\n")
cat("Test set size:", nrow(test_set), "\n")

# 3. EXPLORATORY DATA ANALYSIS (PLOTS)
# ===============================================================================

# Re-load or use unscaled data for clearer EDA interpretation if needed, 
# but we'll use the original dataframe 'dataset' for initial distribution plots 
# before scaling for the model.

# PLOT 1: Age Distribution by Purchase Status
p1 <- ggplot(dataset, aes(x = Age, fill = Purchased)) +
  geom_density(alpha = 0.5) +
  labs(title = "Age Distribution by Purchase Status",
       x = "Age", y = "Density", fill = "Purchased") +
  theme_minimal()
ggsave("plot1_age_distribution.png", p1, width = 8, height = 6)

# PLOT 2: Estimated Salary Boxplot
p2 <- ggplot(dataset, aes(x = Purchased, y = EstimatedSalary, fill = Purchased)) +
  geom_boxplot() +
  labs(title = "Estimated Salary by Purchase Status",
       x = "Purchased", y = "Estimated Salary") +
  theme_minimal()
ggsave("plot2_salary_distribution.png", p2, width = 8, height = 6)

# PLOT 3: Age vs Salary Scatter Plot
p3 <- ggplot(dataset, aes(x = Age, y = EstimatedSalary, color = Purchased)) +
  geom_point(alpha = 0.7, size = 3) +
  labs(title = "Age vs Estimated Salary",
       x = "Age", y = "Estimated Salary", color = "Purchased") +
  theme_minimal()
ggsave("plot3_age_vs_salary.png", p3, width = 8, height = 6)

# PLOT 4: Gender Count
p4 <- ggplot(dataset, aes(x = Gender, fill = Purchased)) +
  geom_bar(position = "dodge") +
  labs(title = "Purchase Counts by Gender",
       x = "Gender", y = "Count") +
  scale_x_discrete(labels = c("Male", "Female")) +
  theme_minimal()
ggsave("plot4_gender_count.png", p4, width = 8, height = 6)

cat("\nEDA Plots 1-4 saved.\n")


# 4. LOGISTIC REGRESSION MODEL
# ===============================================================================
# Fitting Logistic Regression to the Training set
classifier <- glm(formula = Purchased ~ .,
                  family = binomial,
                  data = training_set)

cat("\n=== MODEL SUMMARY ===\n")
print(summary(classifier))

# Predicting the Test set results
prob_pred <- predict(classifier, type = 'response', newdata = test_set[-4])
y_pred <- ifelse(prob_pred > 0.5, 1, 0)
y_pred_factor <- factor(y_pred, levels = c(0, 1))


# 5. EVALUATION
# ===============================================================================
# Confusion Matrix
cm <- table(test_set[, 4], y_pred)
cat("\n=== CONFUSION MATRIX ===\n")
print(cm)

# Accuracy
accuracy <- sum(diag(cm)) / sum(cm)
cat("\nAccuracy:", round(accuracy * 100, 2), "%\n")

# PLOT 5: ROC Curve
png("plot5_roc_curve.png", width = 800, height = 600)
pred_obj <- prediction(prob_pred, test_set$Purchased)
perf <- performance(pred_obj, measure = "tpr", x.measure = "fpr")
plot(perf, col = "blue", lwd = 2, main = "ROC Curve")
abline(a = 0, b = 1, lty = 2, col = "gray")
auc <- performance(pred_obj, measure = "auc")
auc_value <- unlist(auc@y.values)
legend("bottomright", legend = paste("AUC =", round(auc_value, 4)), col = "blue", lwd = 2)
dev.off()

cat("ROC Curve saved as plot5_roc_curve.png\n")


# 6. VISUALISING THE TRAINING SET RESULTS
# ===============================================================================
# Note: This visualization is computationally intensive for the background grid
png("plot6_decision_boundary.png", width = 1000, height = 800)

set = training_set
X1 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05) # Age
X2 = seq(min(set[, 3]) - 1, max(set[, 3]) + 1, by = 2500) # Salary is scaled differently in formula, but careful here.
# Actually, since we scaled the data, the ranges are small (around -2 to 2) 
# Let's adjust ranges for scaled data
X1 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
X2 = seq(min(set[, 3]) - 1, max(set[, 3]) + 1, by = 0.01)

grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')

# We need to add Gender to grid_set to make predictions. 
# We'll assume Gender = 1 (Male) for the boundary visualization or handle it.
# This 2D plot approach works best with only 2 predictors. 
# Since we have Gender, we can't easily plot a 2D boundary unless we fix Gender.
# Simplified approach: Fix Gender to 'Male' (1) for visualization purposes
grid_set$Gender = factor(1, levels = c(1, 2)) 

# Reorder columns to match model input: Gender, Age, EstimatedSalary
grid_set = grid_set[, c(3, 1, 2)]

prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)

plot(set[, 2:3],
     main = 'Logistic Regression (Training set) - Visualized for Males',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
points(grid_set[y_grid == 1, 2:3], col = 'springgreen3', pch = '.')
points(grid_set[y_grid == 0, 2:3], col = 'tomato', pch = '.')
points(set[, 2:3], pch = 21, bg = ifelse(set[, 4] == 1, 'green4', 'red3'))

dev.off()
cat("Decision boundary plot saved as plot6_decision_boundary.png\n")

cat("\nAnalysis Complete!\n")
