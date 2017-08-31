# assignment 1 - isye 6501
# Group Members: Nirmit Chetwani, Mansi Arora, Nabila Usmani, Jeh Lokhande

setwd("E:/isye_6501/week_1_hw")
#install.packages("kernlab")
#install.packages("kknn")
library("kernlab")
library("kknn")
data <- read.table("credit_card_data-headers.txt", header = TRUE)
data_mat <- as.matrix(data)

################## SVM Model Problem ####################

class_lambda_table = data.frame(matrix(nrow = 15, ncol = 4, 0))
colnames(class_lambda_table) = c("lambda", "accuracy", "margin", "num_support_vectors")

# calculate the distance given ai's
margin_svm <- function(a) {
  dist <- round(2 / sqrt(sum(a^2)), 3)
  return(dist)
}

# return 
svm_model_results <- function(data, lambda) {
  model <- ksvm(data[, 1:10], data[, 11], type = "C-svc", kernel = "vanilladot", C = lambda, scaled = TRUE)
  # calculate vector a1, a2, .. , am
  a <- colSums(model@xmatrix[[1]] * model@coef[[1]])
  a0 <- -model@b
  margin <- margin_svm(a)
  num_svs <- model@nSV
  prediction <- predict(model, data[, 1:10])
  accuracy <- sum(prediction == data[, 11]) / nrow(data)
  result <- data.frame(lambda, accuracy, margin, num_svs)
  return(result)
}

for(i in 1:15) { 
  lambda <- 10 ^ (i - 3)
  class_lambda_table[i, ] <- svm_model_results(data_mat, lambda)
  print(i)
}

plot(log10(class_lambda_table[1:15, 1]), class_lambda_table[1:15, 2], xlab = "logb 10 of lambda values", ylab = "accuracy")
plot(log10(class_lambda_table[1:15, 1]), class_lambda_table[1:15, 3], xlab = "logb 10 of lambda values", ylab = "margin")
plot(log10(class_lambda_table[1:15, 1]), class_lambda_table[1:15, 4], xlab = "logb 10 of lambda values", ylab = "no of sv's")


################## KNN Model Problem  - Final Solution (a) ####################

acc_against_k_fin <- data.frame(matrix(nrow = 20, ncol = 2, 0))
colnames(acc_against_k_fin) <- c("k_value", "accuracy")
for(j in 1:20) {
  sum <- 0
  for(i in 1:nrow(data)) { 
    train_data <- data[-i, ]
    test_data <- data[i, ]
    # building different model for every data point; so in nut shell, we will be building 654*20 models in this case
    model <- kknn(formula = R1 ~. , train_data, test_data, k = j , scale = TRUE)
    predicted <- fitted(model)
    if (abs(data[i, 11] - predicted) < 0.5) {
      sum = sum + 1
    }
  }
  accuracy = sum / nrow(data)
  acc_against_k_fin[j, 1] <- j
  acc_against_k_fin[j, 2] <- accuracy
  print(j)
}
plot(acc_against_k_fin[1:20, 1], acc_against_k_fin[1:20, 2], xlim = c(1, 20), ylim = c(0.7, 0.9))


################## KNN Model Problem  - Alternate solution (c) ####################

# unpack operator := for unpacking multiple outputs from a function
':=' <- function(lhs, rhs) {
  frame <- parent.frame()
  lhs <- as.list(substitute(lhs))
  if (length(lhs) > 1)
    lhs <- lhs[-1]
  if (length(lhs) == 1) {
    do.call(`=`, list(lhs[[1]], rhs), envir=frame)
    return(invisible(NULL)) 
  }
  if (is.function(rhs) || is(rhs, 'formula'))
    rhs <- list(rhs)
  if (length(lhs) > length(rhs))
    rhs <- c(rhs, rep(list(NULL), length(lhs) - length(rhs)))
  for (i in 1:length(lhs))
    do.call(`=`, list(lhs[[i]], rhs[[i]]), envir=frame)
  return(invisible(NULL)) 
}

# random sampling of data - the ratio of training to testing set is 3:1
random_sample_data <- function(d) {
  rand_train <- data[sample(1:nrow(d), round(0.75*nrow(d), 0), replace = FALSE), ]
  rows <- as.numeric(rownames(rand_train))
  rand_test <- d[-rows, ]
  return(list(rand_train, rand_test))
}

#rotation sampling of data - the ratio of training to testing set is 3:1
rotation_sample_data <- function(d) {
  rot_train <- d[1, ]
  rot_test <- d[2, ]
  for(i in 3:nrow(d)) {
    if (i %% 4 != 0) {
      rot_train <- rbind.data.frame(rot_train, d[i, ])
    } 
    else {
      rot_test <- rbind.data.frame(rot_test, d[i, ])
    }
  }
  return(list(rot_train, rot_test))
}

# assigning training and testing data from both methodologies to data frames
c(rd_sample_train, rd_sample_test) := random_sample_data(data)
c(rotation_train, rotation_test) := rotation_sample_data(data)

accuracy_against_k <- data.frame(matrix(nrow = 20, ncol = 3, 0))
colnames(accuracy_against_k) <- c("k_value", "random_sampling_accuracy", "rotation_sampling_accuracy")

knn_model_results <- function(k_value) {
  rand_model <- kknn(formula = R1 ~. , rd_sample_train, rd_sample_test, k = k_value , scale = TRUE)
  pred_rand <- fitted(rand_model)
  pred_rand <- ifelse(pred_rand > 0.5, 1, 0)
  rand_accuracy <- sum(pred_rand == rd_sample_test[, 11]) / nrow(rd_sample_test)
  
  rot_model <- kknn(formula = R1 ~. , rotation_train, rotation_test, k = k_value , scale = TRUE)
  pred_rot <- fitted(rot_model)
  pred_rot <- ifelse(pred_rot > 0.5, 1, 0)
  rot_accuracy <- sum(pred_rot == rotation_test[, 11]) / nrow(rotation_test)
  
  knn_result <- data.frame(rand_accuracy, rot_accuracy)
  return(knn_result)
}

for(i in 1:20) { 
  accuracy_against_k[i, 1] <- i
  accuracy_against_k[i, 2:3] <- knn_model_results(i)
}
plot(accuracy_against_k[1:20, 1], accuracy_against_k[1:20, 2], xlab = "k-value", ylab = "random_sample_accuracy")
plot(accuracy_against_k[1:20, 1], accuracy_against_k[1:20, 3], xlab = "k-value", ylab = "rotation_accuracy")
