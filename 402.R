library(tidyverse)
library(cluster)
library(factoextra)
library(caret)
library(clValid)
library(ggplot2)
library(dendextend)

# Load the dataset
heart_c <- read_csv("C:/Users/USER/Desktop/HDS/machine learning/data/heart-c.csv")

str(heart_c)

# Remove the 'num' attribute as it's the class label not to be included in clustering
# and the '...1' column.
heart_c <- heart_c %>% select(-num, -"...1")

# Remove missing values 
heart_c <- na.omit(heart_c)

# Convert binary (logical) variables to numeric
heart_c$fbs <- as.numeric(heart_c$fbs)

# Convert 'sex' and 'exang' into binary numeric variables
heart_c$sex <- ifelse(heart_c$sex == "male", 1, 0)
heart_c$exang <- ifelse(heart_c$exang == "yes", 1, 0)

# Convert to factors
heart_c$cp <- factor(heart_c$cp, levels = c("typ_angina", "atyp_angina", "non_anginal", "asympt"))
heart_c$restecg <- factor(heart_c$restecg, levels = c("normal", "st_t_wave_abnormality", "left_vent_hyper"))
heart_c$slope <- factor(heart_c$slope, levels = c("up", "flat", "down"))
heart_c$thal <- factor(heart_c$thal, levels = c("normal", "fixed_defect", "reversable_defect"))

# One-hot encoding 
dummy_model <- dummyVars("~ . ", data = heart_c)
heart_c_transformed <- predict(dummy_model, newdata = heart_c)


# Normalise the data
heart_c_scaled <- scale(heart_c_transformed)

# Initialize total within sum of squares error: wss
wss <- 0

# For 1 to 5 cluster centers
for (i in 1:10) {
  km_out <- kmeans(heart_c_scaled, centers = i, nstart = 20)
  # Save total within sum of squares to wss variable
  wss[i] <- km_out$tot.withinss
}

# Plot total within sum of squares vs. number of clusters
plot(1:10, wss, type = "b", 
     xlab = "Number of Clusters", 
     ylab = "Within groups sum of squares")

# Perform K-means clustering
km_out <- kmeans(heart_c_scaled, centers = 2, nstart = 20)

# Visualise clusters
fviz_cluster(km_out, data = heart_c_scaled)




# Boxplot for visualizing distributions of numeric features
heart_c %>% 
  select_if(is.numeric) %>%
  gather(key = "features", value = "value") %>%
  ggplot(aes(x = features, y = value)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Hierarchal clustering

# Compute the distance matrix
dist_mat <- dist(heart_c_scaled)

# Perform hierarchical clustering with average linkage
hclust_avg <- hclust(dist_mat, method = "average")

# Plot the dendrogram
plot(hclust_avg, main = "Hierarchical Clustering Dendrogram", sub= "", xlab = "")

# Initialize the dunns vector 
dunns <- 0 

# Loop from 1 to 10 to calculate Dunn index for each number of clusters
for (i in 1:10) {
  cluster_assignments <- cutree(hclust_avg, i)
  dunns[i] <- dunn(dist_mat, cluster_assignments)
}

# Plot dunns vector against the cluster numbers 1 to 10
plot(dunns, type = "b", xlab = "Number of clusters", ylab = "Dunn Index",
     main = "Dunn Index for Different Numbers of Clusters")

# The dendrogram at k = 2 and color the clusters
dend <- as.dendrogram(hclust_avg)
dend_colored <- color_branches(dend, k = 2)
plot(dend_colored, main = "Dendrogram with k = 2 Clusters")



