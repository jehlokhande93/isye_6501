# assignment 2 
setwd("E:/isye_6501/week_2_hw")
#install.packages("data.table")
library("kernlab")
library("kknn")
library("data.table")
# general steps for solution 1

data <- read.table("credit_card_data-headers.txt", header = TRUE)
data_mat <- as.matrix(data)
# solution 1 -a : knn using k-fold cross validation
# ENTER THE K -Value HERE
k_value = 5

# random sampling of data: using k-fold approach. this function returns k datasets randomly sampled from the parent data 
random_sample_data_k_fold <- function(d, k) {
  rand_data <- data
  result <- list()
  for(i in 1:(k)) {
    rand_train_temp <- data[sample(1:nrow(rand_data), round((1/(k-i+2))*nrow(rand_data), 0), replace = FALSE), ]
    name <- paste("rand_data_part", i, sep = "_") 
    assign(name, rand_train_temp)
    rows <- as.numeric(rownames(rand_train_temp))
    rand_data <- rand_data[-rows, ]
    result[[i]] <- rand_train_temp
  }
  result[[k+1]] <- rand_data
  return(result)
}
list_random_sample <- random_sample_data_k_fold(data, k_value)

#rotation sampling of data - the ratio of training to testing set is 3:1
rotation_sample_data_k_fold <- function(d, k) {
  result <- list()
  for(i in 1:(k+1)) {
    temp <- seq(i, nrow(d), k+1)
    rot_sample <- d[temp, ]
    result[[i]] <- rot_sample
  }
  return(result)
}
list_rotation_sample <- rotation_sample_data_k_fold(data, k_value)

rand_val <- list()
rand_train <- list()
rot_val <- list()
rot_train <- list()
rand_test <- list_random_sample[[k_value+1]]
rot_test <- list_rotation_sample[[k_value+1]]
for(i in 1:k_value) {
  rand <- data.frame(matrix(nrow = 0, ncol = 11))
  colnames(rand) <- colnames(data)
  rotation <- data.frame(matrix(nrow = 0, ncol = 11))
  colnames(rotation) <- colnames(data)
  for(j in 1:k_value) {
    if(j == i) {
      rand_val[[i]] <- list_random_sample[[j]]
      rot_val[[i]] <- list_rotation_sample[[j]]
    }
    else {
      rand <- rbind.data.frame(rand, list_random_sample[[j]])
      rotation <- rbind.data.frame(rotation, list_rotation_sample[[j]])
    }
  }
  rand_train[[i]] <- rand
  rot_train[[i]] <- rotation
}

knn_function_cross_validation <- function(rotation_train, rotation_val, random_train, random_val, k, max_k_value) {
  accuracy_knn_data_frame <- data.frame(matrix(nrow = max_k_value*k , ncol = 4, 0))
  colnames(accuracy_knn_data_frame) <- c("k_knn_value", "index_value", "rotation_accuracy", "random_accuracy")
  p <- 1
  for(r in 1:max_k_value) {
    for(i in 1:k) {
      model_random <- kknn(formula = R1 ~. , random_train[[i]], random_val[[i]], k = r , scale = TRUE)
      model_rotation <- kknn(formula = R1 ~. , rotation_train[[i]], rotation_val[[i]], k = r , scale = TRUE)
      pred_random <- fitted(model_random)
      pred_rotation <- fitted(model_rotation)
      pred_rand_acc <- ifelse(pred_random > 0.5, 1, 0)
      rand_accuracy <- sum(pred_rand_acc == random_val[[i]][, 11]) / nrow(random_val[[i]])
      pred_rot_acc <- ifelse(pred_rotation > 0.5, 1, 0)
      rotation_accuracy <- sum(pred_rot_acc == rotation_val[[i]][, 11]) / nrow(rotation_val[[i]])
      accuracy_knn_data_frame[p, 1] <- r
      accuracy_knn_data_frame[p, 2] <- i
      accuracy_knn_data_frame[p, 3] <- rotation_accuracy
      accuracy_knn_data_frame[p, 4] <- rand_accuracy
      p <- p + 1
    }
  }
  return(accuracy_knn_data_frame)
}
output_knn <- knn_function_cross_validation(rot_train, rot_val, rand_train, rand_val, k_value, 20)
output_knn_table <- data.table(output_knn)
aggregated_knn_output <- output_knn_table[,list(mean_rotation_accuracy=mean(rotation_accuracy), mean_random_accuracy=mean(random_accuracy), sd_rotation_accuracy=sd(rotation_accuracy), sd_random_accuracy=sd(random_accuracy)),by=k_knn_value]

# svm model
margin_svm <- function(a) {
  dist <- round(2 / sqrt(sum(a^2)), 3)
  return(dist)
}

svm_model_results <- function(rotation_train, rotation_val, random_train, random_val, k) {
  p <- 1
  accuracy_svm_data_frame <- data.frame(matrix(nrow = 5*k , ncol = 6, 0))
  colnames(accuracy_svm_data_frame) <- c("lambda_val", "index_value", "rotation_accuracy", "random_accuracy", "rotation_margin", "random_margin")
  for(r in 1:5) {
    lambda = 10^(r - 1)
    for(i in 1:k) {
      model_rand <- ksvm(as.matrix(random_train[[i]][, 1:10]), as.matrix(random_train[[i]][, 11]), type = "C-svc", kernel = "vanilladot", C = lambda, scaled = TRUE)
      model_rot <- ksvm(as.matrix(rotation_train[[i]][, 1:10]), as.matrix(rotation_train[[i]][, 11]), type = "C-svc", kernel = "vanilladot", C = lambda, scaled = TRUE)
      # calculate vector a1, a2, .. , am
      a_rand <- colSums(model_rand@xmatrix[[1]] * model_rand@coef[[1]])
      a_rot <- colSums(model_rot@xmatrix[[1]] * model_rot@coef[[1]])
      margin_rand <- margin_svm(a_rand)
      margin_rot <- margin_svm(a_rot)
      num_svs_rand <- model_rand@nSV
      num_svs_rot <- model_rot@nSV
      prediction_rand <- predict(model_rand, random_val[[i]][, 1:10])
      accuracy_rand <- sum(prediction_rand == random_val[[i]][, 11]) / nrow(random_val[[i]])
      prediction_rot <- predict(model_rot, rotation_val[[i]][, 1:10])
      accuracy_rot <- sum(prediction_rot == rotation_val[[i]][, 11]) / nrow(rotation_val[[i]])
      accuracy_svm_data_frame[p, 1] <- lambda
      accuracy_svm_data_frame[p, 2] <- i
      accuracy_svm_data_frame[p, 3] <- accuracy_rand
      accuracy_svm_data_frame[p, 4] <- accuracy_rot
      accuracy_svm_data_frame[p, 5] <- margin_rand
      accuracy_svm_data_frame[p, 6] <- margin_rot
      p <- p + 1
    }
    print(r)
  }
  return(accuracy_svm_data_frame)
}

output_svm <- svm_model_results(rot_train, rot_val, rand_train, rand_val, k_value)
output_svm_table <- data.table(output_svm)
aggregated_svm_output <- output_svm_table[,list(mean_rotation_accuracy=mean(rotation_accuracy), mean_random_accuracy=mean(random_accuracy), sd_rotation_accuracy=sd(rotation_accuracy), sd_random_accuracy=sd(random_accuracy)),by=lambda]