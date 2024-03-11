newst<-read.transactions("news_transactions.csv",
                         rm.duplicates = FALSE,
                         format = "basket",
                         sep = ",")
redditt<-read.transactions("reddit_transactions.csv",
                         rm.duplicates = FALSE,
                         format = "basket",
                         sep = ",")
mediumt<-read.transactions("medium_transactions.csv",
                         rm.duplicates = FALSE,
                         format = "basket",
                         sep = ",")

newsrules = arules::apriori(newst,parameter = list(support=.02,
                                                   confidence=.5,
                                                   minlen=2))
redditrules = arules::apriori(redditt,parameter = list(support=.01,
                                                       confidence=.5,
                                                       minlen=2))
mediumrules = arules::apriori(mediumt,parameter = list(support=.02,
                                                       confidence=.5,
                                                       minlen=2))


itemFrequencyPlot(newst, topN=20, type="absolute")
itemFrequencyPlot(redditt, topN=20, type="absolute")
itemFrequencyPlot(mediumt, topN=20, type="absolute")


sortednews <- sort(newsrules, by="confidence", decreasing=TRUE)
inspect(sortednews[1:15])
(summary(sortednews))

sortedreddit <- sort(redditrules, by="confidence", decreasing=TRUE)
inspect(sortedreddit[1:15])
(summary(sortedreddit))

sortedmedium <- sort(mediumrules, by="confidence", decreasing=TRUE)
inspect(sortedmedium[1:15])
(summary(sortedmedium))
#########################

sortednews <- sort(newsrules, by="support", decreasing=TRUE)
inspect(sortednews[1:15])
(summary(sortednews))

sortedreddit <- sort(redditrules, by="support", decreasing=TRUE)
inspect(sortedreddit[1:15])
(summary(sortedreddit))

sortedmedium <- sort(mediumrules, by="support", decreasing=TRUE)
inspect(sortedmedium[1:15])
(summary(sortedmedium))
#########################

sortednews <- sort(newsrules, by="lift", decreasing=TRUE)
inspect(sortednews[1:15])
(summary(sortednews))

sortedreddit <- sort(redditrules, by="lift", decreasing=TRUE)
inspect(sortedreddit[1:15])
(summary(sortedreddit))

sortedmedium <- sort(mediumrules, by="lift", decreasing=TRUE)
inspect(sortedmedium[1:15])
(summary(sortedmedium))



plot(newsrules, method="graph", control=list(type="items"))


rules_to_visNetwork <- function(rules) {
  lhs <- labels(lhs(rules))
  rhs <- labels(rhs(rules))
  support <- quality(rules)$support
  confidence <- quality(rules)$confidence
  lift <- quality(rules)$lift
  
  edges <- data.frame(from = lhs, to = rhs, support = support, confidence = confidence, lift = lift)
  
  nodes <- data.frame(id = unique(c(lhs, rhs)), label = unique(c(lhs, rhs)))
  
  list(edges = edges, nodes = nodes)
}

news_viz_data <- rules_to_visNetwork(newsrules)

visNetwork(nodes = news_viz_data$nodes, edges = news_viz_data$edges) %>%
  visEdges(arrows = 'to') %>%
  visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE) %>%
  visLayout(randomSeed = 123)

