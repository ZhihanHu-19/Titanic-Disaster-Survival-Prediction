# Step 1: Load data
cat("\nStep 1: Load data\n")
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

cat("Train shape:", nrow(train), "x", ncol(train), "\n")
cat("Test shape:", nrow(test), "x", ncol(test), "\n")

# Step 2: Process data
cat("\nStep 2: Process data\n")

# fill values
train$Age[is.na(train$Age)] <- median(train$Age, na.rm = TRUE)
test$Age[is.na(test$Age)] <- median(test$Age, na.rm = TRUE)
train$Embarked[train$Embarked == ""] <- "S"
test$Fare[is.na(test$Fare)] <- median(test$Fare, na.rm = TRUE)

# categorical factors
train$Sex <- factor(train$Sex)
test$Sex <- factor(test$Sex, levels=levels(train$Sex))
train$Embarked <- factor(train$Embarked)
test$Embarked <- factor(test$Embarked, levels = levels(train$Embarked))

# Select features
features <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")
formula <- as.formula(paste("Survived ~", paste(features, collapse = " + ")))

# Step 3: Train the model
cat("\nStep 3: Train the model\n")
model <- glm(formula, data=train, family=binomial)

# training dataset predictions
train$pred <- ifelse(predict(model, train, type = "response") > 0.5, 1, 0)
train_accuracy <- mean(train$pred == train$Survived)
cat("Training accuracy:", round(train_accuracy, 4), "\n")

# Step 4: Predict test
cat("\nStep 4: Predict test\n")
test$Survived <- ifelse(predict(model, test, type = "response") > 0.5, 1, 0)
cat("First 10 test predictions:", head(test$Survived, 10), "\n")

# Step 5: Save submission
cat("\nStep 5: Save predictions\n")
write.csv(test[, c("PassengerId", "Survived")], "data/predictions_r.csv", row.names = FALSE)
cat("Saved predictions to data/predictions_r.csv\n")