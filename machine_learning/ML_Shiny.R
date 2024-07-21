library(shiny)
library(caret)
library(lattice)

# Load and prepare the dataset
data(iris)
dataset <- iris
validation_index <- createDataPartition(dataset$Species, p = 0.80, list = FALSE)
validation <- dataset[-validation_index,]
dataset <- dataset[validation_index,]

x <- dataset[, 1:4]
y <- dataset[, 5]

# Shiny UI
ui <- fluidPage(
  titlePanel("Iris Dataset Analysis and Model Comparison"),
  sidebarLayout(
    sidebarPanel(
      helpText("This app demonstrates various machine learning models on the Iris dataset.")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Boxplot", plotOutput("boxPlot")),
        tabPanel("Barplot", plotOutput("barPlot")),
        tabPanel("Scatterplot Matrix", plotOutput("scatterMatrix")),
        tabPanel("Model Training", verbatimTextOutput("modelSummary")),
        tabPanel("Model Comparison", plotOutput("modelComparison")),
        tabPanel("Best Model", verbatimTextOutput("bestModel")),
        tabPanel("Confusion Matrix", verbatimTextOutput("confusionMatrix"))
      )
    )
  )
)

# Shiny Server
server <- function(input, output) {
  output$boxPlot <- renderPlot({
    par(mfrow = c(1, 4), mar = c(4, 4, 2, 1))
    for (i in 1:4) {
      boxplot(x[, i], main = names(iris)[i])
    }
  })
  
  output$barPlot <- renderPlot({
    plot(y)
  })
  
  output$scatterMatrix <- renderPlot({
    pairs(x, col = as.numeric(y), pch = 19, main = "Scatterplot Matrix")
  })
  
  output$modelSummary <- renderPrint({
    control <- trainControl(method = "cv", number = 10)
    metric <- "Accuracy"
    
    set.seed(7)
    fit.lda <- train(Species ~ ., data = dataset, method = "lda", metric = metric, trControl = control)
    
    set.seed(7)
    fit.cart <- train(Species ~ ., data = dataset, method = "rpart", metric = metric, trControl = control)
    
    set.seed(7)
    fit.knn <- train(Species ~ ., data = dataset, method = "knn", metric = metric, trControl = control)
    
    set.seed(7)
    fit.svm <- train(Species ~ ., data = dataset, method = "svmRadial", metric = metric, trControl = control)
    
    set.seed(7)
    fit.rf <- train(Species ~ ., data = dataset, method = "rf", metric = metric, trControl = control)
    
    results <- resamples(list(lda = fit.lda, cart = fit.cart, knn = fit.knn, svm = fit.svm, rf = fit.rf))
    summary(results)
  })
  
  output$modelComparison <- renderPlot({
    control <- trainControl(method = "cv", number = 10)
    metric <- "Accuracy"
    
    set.seed(7)
    fit.lda <- train(Species ~ ., data = dataset, method = "lda", metric = metric, trControl = control)
    
    set.seed(7)
    fit.cart <- train(Species ~ ., data = dataset, method = "rpart", metric = metric, trControl = control)
    
    set.seed(7)
    fit.knn <- train(Species ~ ., data = dataset, method = "knn", metric = metric, trControl = control)
    
    set.seed(7)
    fit.svm <- train(Species ~ ., data = dataset, method = "svmRadial", metric = metric, trControl = control)
    
    set.seed(7)
    fit.rf <- train(Species ~ ., data = dataset, method = "rf", metric = metric, trControl = control)
    
    results <- resamples(list(lda = fit.lda, cart = fit.cart, knn = fit.knn, svm = fit.svm, rf = fit.rf))
    dotplot(results)
  })
  
  output$bestModel <- renderPrint({
    control <- trainControl(method = "cv", number = 10)
    metric <- "Accuracy"
    
    set.seed(7)
    fit.lda <- train(Species ~ ., data = dataset, method = "lda", metric = metric, trControl = control)
    
    print(fit.lda)
  })
  
  output$confusionMatrix <- renderPrint({
    control <- trainControl(method = "cv", number = 10)
    metric <- "Accuracy"
    
    set.seed(7)
    fit.lda <- train(Species ~ ., data = dataset, method = "lda", metric = metric, trControl = control)
    
    predictions <- predict(fit.lda, validation)
    confusionMatrix(predictions, validation$Species)
  })
}

# Run the application 
shinyApp(ui = ui, server = server)