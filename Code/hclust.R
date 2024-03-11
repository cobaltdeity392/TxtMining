news_lem<-read.csv("title_countvec_lem.csv")
reddit_lem<-read.csv("reddit_countvec_lem.csv")
medium_lem<-read.csv("medium_countvec_lem_title.csv")


distMatrix_C <- as.dist(proxy::simil(news_lem, method = "cosine"))
distMatrix_C <- 1 - distMatrix_C
print("cos sim matrix is :\n")
print(distMatrix_C) ##small number is less distant
(HClust_news<- hclust(distMatrix_C, 
                     method = "ward.D"))
plot(HClust_news, cex=0.7, hang=-30, 
     main = "NewsAPI Cosine Sim k=6",labels=FALSE)
rect.hclust(HClust_news, k=6)
(HClust_news<- hclust(distMatrix_C, 
                  method = "ward.D"))
plot(HClust_news, cex=0.7, hang=-30, 
     main = "NewsAPI Cosine Sim k=5",labels=FALSE)
rect.hclust(HClust_news, k=5)
(HClust_news<- hclust(distMatrix_C, 
                  method = "ward.D"))
plot(HClust_news, cex=0.7, hang=-30, 
     main = "NewsAPI Cosine Sim k=7",labels=FALSE)
rect.hclust(HClust_news, k=7)


distMatrix_C <- as.dist(proxy::simil(reddit_lem, method = "cosine"))
distMatrix_C <- 1 - distMatrix_C
print("cos sim matrix is :\n")
print(distMatrix_C) ##small number is less distant
(HClust_reddit<- hclust(distMatrix_C, 
                  method = "ward.D"))
plot(HClust_reddit, cex=0.7, hang=-30, 
     main = "Reddit Cosine Sim k=4",labels=FALSE)
rect.hclust(HClust_reddit, k=4)
(HClust_reddit<- hclust(distMatrix_C, 
                    method = "ward.D"))
plot(HClust_reddit, cex=0.7, hang=-30, 
     main = "Reddit Cosine Sim k=3",labels=FALSE)
rect.hclust(HClust_reddit, k=3)
(HClust_reddit<- hclust(distMatrix_C, 
                    method = "ward.D"))
plot(HClust_reddit, cex=0.7, hang=-30, 
     main = "Reddit Cosine Sim k=5",labels=FALSE)
rect.hclust(HClust_reddit, k=5)

distMatrix_C <- as.dist(proxy::simil(medium_lem, method = "cosine"))
distMatrix_C <- 1 - distMatrix_C
print("cos sim matrix is :\n")
print(distMatrix_C) ##small number is less distant
(HClust_medium<- hclust(distMatrix_C, 
                    method = "ward.D"))
plot(HClust_medium, cex=0.7, hang=-30, 
     main = "Medium Cosine Sim k=4",labels=FALSE)
rect.hclust(HClust_medium, k=4)
(HClust_medium<- hclust(distMatrix_C, 
                    method = "ward.D"))
plot(HClust_medium, cex=0.7, hang=-30, 
     main = "Medium Cosine Sim k=3",labels=FALSE)
rect.hclust(HClust_medium, k=3)
(HClust_medium<- hclust(distMatrix_C, 
                    method = "ward.D"))
plot(HClust_medium, cex=0.7, hang=-30, 
     main = "Medium Cosine Sim k=2",labels=FALSE)
rect.hclust(HClust_medium, k=2)

