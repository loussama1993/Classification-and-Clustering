library(tidyverse)
library(randomForest)
library(caret)
library(pROC)


modified_data <- read_csv("C:/Users/USER/Desktop/HDS/machine learning/data/heart_disease_modified.csv")
sum(is.na(modified_data))
str(modified_data)

modified_data <- modified_data %>% select(-Patient_ID, -"...1",-pace_maker, 
                                          -perfusion, -traponin)

# Recoding 'cp'
modified_data$cp <- factor(modified_data$cp, levels = c(1, 2, 3, 4), labels = c("typ_angina", "atyp_angina", "non_anginal", "asympt"))

# Recoding 'restecg' 
modified_data$restecg <- factor(modified_data$restecg, levels = c(0, 1, 2), labels = c("normal", "st_t_wave_abnormality", "left_vent_hyper"))

# Recoding 'slope' 
modified_data$slope <- factor(modified_data$slope, levels = c(1, 2, 3), labels = c("up", "flat", "down"))

# Recoding 'thal' 
modified_data$thal <- factor(modified_data$thal, levels = c(3, 6, 7), labels = c("normal", "fixed_defect", "reversable_defect"))

# Recoding 'drug' 
modified_data$drug <- factor(modified_data$drug, levels = c("Aspirin", "Both", "Clopidogrel", "None"))

# Recoding 'fam_hist' (family history of heart disease)
modified_data$fam_hist <- ifelse(modified_data$fam_hist== "yes", 1, 0)

# Identify continuous features
continuous_features <- c("age", "trestbps", "chol", "thalach", "oldpeak", "ca") 

# Scale continuous features only
modified_data[continuous_features] <- scale(modified_data[continuous_features])

# One-hot encoding
dummy_model <- dummyVars("~ .", data = modified_data, fullRank = TRUE)
modified_data_transformed <- predict(dummy_model, newdata = modified_data)

data_transformed <- as.data.frame(modified_data_transformed)

data_transformed$class <- factor(data_transformed$class, levels=c("0", "1"))


set.seed(123)


# Calculate indices for splitting
total_rows <- nrow(data_transformed)
train_rows <- round(0.75 * total_rows)
val_rows <- round(0.15 * total_rows)
# The remaining rows go to the test set
test_rows <- total_rows - train_rows - val_rows

# Create random indices
indices <- sample(1:total_rows)

# Split data based on calculated indices
trainset <- data_transformed[indices[1:train_rows], ]
valset <- data_transformed[indices[(train_rows + 1):(train_rows + val_rows)], ]
testset <- data_transformed[indices[(train_rows + val_rows + 1):total_rows], ]

# Verify the sizes
print(nrow(trainset))
print(nrow(valset))
print(nrow(testset))

# Logistic Regression
model_log <- glm(class ~ ., data = trainset, family = binomial)
predictions_log <- predict(model_log, valset, type = "response")

# Convert probabilities to binary outcome
predictedClass_log <- ifelse(predictions_log > 0.5, 1, 0)

# Evaluation
predictedClass_log_factor <- factor(predictedClass_log, levels=c("0", "1"))
confusion_log <- confusionMatrix(data=predictedClass_log_factor, reference=valset$class, positive="1")
sensitivity_log <- confusion_log$byClass["Sensitivity"]
sensitivity_log

confusionMatrix_log <- table(Predicted = predictedClass_log_factor, Actual = valset$class)
confusionMatrix_log

# Random Forest
model_rf <- randomForest(class ~ ., data = trainset)
predictions_rf <- predict(model_rf, valset)



# Evaluation

predictions_rf_factor <- factor(predictions_rf, levels=c("0", "1"))
confusion_rf <- confusionMatrix(data=predictions_rf_factor, reference=valset$class, positive="1")
sensitivity_rf <- confusion_rf$byClass["Sensitivity"]
sensitivity_rf

confusionMatrix_rf <- table(Predicted = predictions_rf_factor, Actual = valset$class)
confusionMatrix_rf

# ROC Curve for Logistic Regression
roc_log <- roc(valset$class, predictions_log)
plot(roc_log, main="ROC Curve for Logistic Regression")

# ROC Curve for Random Forest
predictions_rf_prob <- predict(model_rf, valset, type="prob")[,2] # Get probabilities for class=1
roc_rf <- roc(valset$class, predictions_rf_prob)
plot(roc_rf, main="ROC Curve for Random Forest")

# AUC to compare the models
auc_log <- auc(roc_log)
auc_rf <- auc(roc_rf)
auc_log
auc_rf


table(modified_data$slope)
importance(model_rf)
varImpPlot(model_rf)

# Specifying the control using k-fold cross-validation
trainset$class <- factor(trainset$class, levels = c("1", "0"), labels = c("Positive", "Negative"))

control <- trainControl(method = "repeatedcv",
                        number = 10,
                        repeats = 3,
                        summaryFunction = twoClassSummary,  
                        classProbs = TRUE,  
                        savePredictions = "final")
# Setting up the train function for Random Forest with k-fold cross-validation
rfModel <- train(class ~ ., data = trainset,
                 method = "rf",
                 trControl = control,
                 metric = "Sensitivity")

# For sensitivity analysis post-modeling
testset$class <- factor(testset$class, levels = c("1", "0"), labels = c("Positive", "Negative"))

predictions2 <- predict(rfModel, newdata = testset)

confMatrix <- confusionMatrix(predictions2, testset$class)

sensitivity2 <- confMatrix$byClass['Sensitivity']

print(sensitivity2)

predictions_prob <- predict(rfModel, newdata = testset, type = "prob")[,2]

rocResult1 <- roc(response = testset$class, predictor = predictions_prob)
plot(rocResult1)
auc(rocResult1)


# Specifying the control using k-fold cross-validation for hyperparameter tuning
# using grid search

control1 <- trainControl(method = "repeatedcv",
                        number = 10,
                        repeats = 3,
                        search = "grid",
                        summaryFunction = twoClassSummary,  
                        classProbs = TRUE,  
                        savePredictions = "final")

# Defining a tuning grid
tunegrid <- expand.grid(.mtry=c(1:15))

# Retraining with hyperparameter tuning
rfTuned <- train(class ~ ., data = trainset,
                 method = "rf",
                 trControl = control1,
                 tuneGrid = tunegrid,
                 metric = "Sensitivity") 

print(rfTuned)

predictions3 <- predict(rfTuned, newdata = testset)

confMatrix2 <- confusionMatrix(predictions3, testset$class)

sensitivity3 <- confMatrix2$byClass['Sensitivity']

print(sensitivity3)


pred_rf_prob <- predict(rfTuned, testset, type="prob")[,2] 

rocResult <- roc(response = testset$class, predictor = pred_rf_prob)

plot(rocResult)

auc(rocResult)


# Define a vector of ntree values you want to explore
ntree_values <- seq(100, 1000, by=100)

# Initialize a vector to store the sensitivity for each ntree value
sensitivities <- numeric(length(ntree_values))

# Loop over ntree values, train a model for each, and evaluate its performance
for (i in seq_along(ntree_values)) {
  # Train the model using the current ntree value
  rfModel <- randomForest(class ~ ., data = trainset, ntree = ntree_values[i])
  
  # Predict on the test set
  predictions <- predict(rfModel, newdata = testset)
  
  # Compute the confusion matrix and extract the sensitivity
  confMatrix <- confusionMatrix(predictions, testset$class)
  sensitivities[i] <- confMatrix$byClass['Sensitivity']
}

# Combine the ntree values and their corresponding sensitivities
results <- data.frame(ntree = ntree_values, Sensitivity = sensitivities)

# Print the results
print(results)