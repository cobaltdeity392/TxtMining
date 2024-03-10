news_lem<-read.csv("title_countvec_lem.csv")
reddit_lem<-read.csv("reddit_countvec_lem.csv")
medium_lem<-read.csv("medium_countvec_lem_title.csv")

(HClust_news<- dist(news_lem, 
                     method = "ward.D"))
plot(HClust_news, cex=0.7, hang=-30, 
     main = "Cosine Sim",labels=FALSE)
rect.hclust(HClust_news, k=6)

(HClust_news<- dist(news_lem, 
                  method = "ward.D"))
plot(HClust_news, cex=0.7, hang=-30, 
     main = "Cosine Sim",labels=FALSE)
rect.hclust(HClust_news, k=5)

(HClust_news<- dist(news_lem, 
                  method = "ward.D"))
plot(HClust_news, cex=0.7, hang=-30, 
     main = "Cosine Sim",labels=FALSE)
rect.hclust(HClust_news, k=7)



(HClust_reddit<- dist(reddit_lem, 
                  method = "ward.D"))
plot(HClust_reddit, cex=0.7, hang=-30, 
     main = "Cosine Sim",labels=FALSE)
rect.hclust(HClust_reddit, k=4)

(HClust_reddit<- dist(reddit_lem, 
                    method = "ward.D"))
plot(HClust_reddit, cex=0.7, hang=-30, 
     main = "Cosine Sim",labels=FALSE)
rect.hclust(HClust_reddit, k=3)

(HClust_reddit<- dist(reddit_lem, 
                    method = "ward.D"))
plot(HClust_reddit, cex=0.7, hang=-30, 
     main = "Cosine Sim",labels=FALSE)
rect.hclust(HClust_reddit, k=5)


(HClust_medium<- dist(medium_lem, 
                    method = "ward.D"))
plot(HClust_medium, cex=0.7, hang=-30, 
     main = "Cosine Sim",labels=FALSE)
rect.hclust(HClust_medium, k=4)

(HClust_medium<- dist(medium_lem, 
                    method = "ward.D"))
plot(HClust_medium, cex=0.7, hang=-30, 
     main = "Cosine Sim",labels=FALSE)
rect.hclust(HClust_medium, k=3)

(HClust_medium<- dist(medium_lem, 
                    method = "ward.D"))
plot(HClust_medium, cex=0.7, hang=-30, 
     main = "Cosine Sim",labels=FALSE)
rect.hclust(HClust_medium, k=2)