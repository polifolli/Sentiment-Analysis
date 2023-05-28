####################################
# Data Professor                   #
# http://youtube.com/dataprofessor #
# http://github.com/dataprofessor  #
####################################


# Import libraries
library(shiny)
library(shinythemes)
library(data.table)
library(RCurl)
library(randomForest)
library(RTextTools)
library(e1071)
library(RWeka)
library(tm)
library(tidytext)
library(dplyr)
library(ggplot2)
library(caret)
library(corpus)
library(quanteda)
library(wordcloud)
library(wordcloud2)
library(extrafont)

#font_import()
#fonts()
# Read data

#setwd("D:/папка/универ/4 курс/ВКР/R/данные/комментарии")
happy = readLines("https://raw.githubusercontent.com/polifolli/Sentiment-Analysis/main/happy.txt", encoding = "UTF-8")
sad = readLines("https://raw.githubusercontent.com/polifolli/Sentiment-Analysis/main/sad.txt", encoding = "UTF-8")

com = c(happy, sad)
com <- gsub("[[:punct:]]", "", com) #удаление пунктуации из строки

sentiment = c(rep("happy", length(happy) ), rep("sad", length(sad)))
com.stem <- text_tokens(com, stemmer = "ru")
toks <- tokens(com.stem, remove_numbers = TRUE, remove_separators = TRUE, remove_symbols = TRUE)
toks <- tokens_remove(toks, c(stopwords('russian'), 'эт', 'ещё'))

dfmat_train <- dfm(toks)

# Build model
sentmod.svm <- svm(x = dfmat_train,
                        y = as.factor(sentiment),
                        kernel = "linear", 
                        cost = 10,  # arbitrary regularization cost
                        probability = TRUE)

# Save model to RDS file
saveRDS(sentmod.svm, "model_svm.rds")

# Read in the SVM model
model <- readRDS("model_svm.rds")

####################################
# User interface                   #
####################################

ui <- fluidPage(
  tags$head(tags$style("

                     #plot{height:1500px !important;}

                     ")),
  theme = shinytheme("yeti"),
  navbarPage(
    "Анализ тональности",
    tabPanel("О проекте",
             sidebarPanel(
               HTML("<h3>Анализ Тональности</h3>"),
               br(),
               p('Лазукова П.И.'),
               p('БИ-19-2'),
               p('НИУ ВШЭ Пермь')
             ),
             mainPanel(
               h2('О проекте'),
               p('Выпускная квалификационная работа на тему: 
                 «Анализ тональности текстов пользовательского контента социальных сетей
                 с помощью методов машинного обучения». '),
               p('Даная работа направлена на проведение анализа тональности пользовательского контента социальных сетей с помощью алгоритмов машинного обучения.
                 Текстовые данные собраны с публичных страниц социальной сети ВКонтакте. 
                 Модели созданы с использованием языка программирования R. '),
               p('Приложение создано в демонстрационных целях.')
             )),
    tabPanel("Облако слов", 
             sidebarPanel(
               HTML("<h3>Об облаках</h3>"),
               p("Облако слов - метод визуализации, который демонстрирует частотность появления слов в определенном тексте."),
               br(),
               p("Данное облако слов построено на основе обучающего множества, которое состоит из комментариев, собранных из социальной сети Вконтакте."),
               HTML("<p>Набор данных находится в открытом доступе <a href='https://github.com/polifolli/Sentiment-Analysis'>по ссылке</a></p>"),
               ),
             mainPanel(
               h2('Облако слов'),
               wordcloud2Output("wordcloud", width = "auto"))
             ),
    tabPanel("Проверка предложения",
             sidebarPanel(
               HTML("<h3>Проверка предложения</h3>"),
               textInput("sentence", label = "Введите предложение на русском языке"), 
               actionButton("submitbutton", label = "Ввод", class = "btn btn-primary")
                         ), #sidebarPanel
             mainPanel(
               h2('Результат анализа'), # Status/Output Text Box
               p("Данная страница демонстрирует результат работы модели анализа тональности, 
                 обученной на наборе данных из", span("1000", style = "font-weight:bold"), "комментариев из социальной сети Вконтакте."),
               p("Для обучения модели был применён метод опорных векторов."),
               h3('Статус'), # Status/Output Text Box
               verbatimTextOutput('contents'),
               h3('Результат'), # Status/Output Text Box
               textOutput('textout')
               #tableOutput('tabledata') # Prediction results table
               
             ) #mainPanel
             ), # Navbar 2, tabPanel 
    tabPanel("SVM-модель",
             # sidebarPanel(
             #   HTML("<h3>Коэффициенты модели</h3>"),
             #   p("оооо"),
             # ),
             h2('Коэффициенты SVM-модели'),
             p("В результате обучения SVM-модель оценивает входящие в обучающее множество слова и присваивает им коэффициенты. Данный график
               иллюстрирует эти коэффициенты, благодаря чему можно наглядно оценить результат обучения модели. Слова имеют особую форму, поскольку
               к текстам применялись алгоритмы стемминга."),
             p("К положительным словам модель отнесла такие слова, как", span("красота, хорошо, добро, прекрасно, люблю, браво, круто, детство, праздник.", style = "font-weight:bold")),
             p("К отрицательным словам модель отнесла такие слова, как", span("жаль, негативно, помойка, мерзость, бред, странно, тупость, бедный.", style = "font-weight:bold")),
             hr(),
             mainPanel(
              
               plotOutput("plot"), width = "100%")
      ),


  ), #navbarPage

) #fluidPage

####################################
# Server                           #
####################################

server <- function(input, output, session) {

  # Input Data
  sentenceInput <- reactive({  

    linesent <- input$sentence
    linesent <- gsub("[[:punct:]]", "", tolower(linesent)) 
    linesent <- text_tokens(linesent, stemmer = "ru")
    line.toks <- tokens(linesent, remove_numbers = TRUE, remove_separators = TRUE, remove_symbols = TRUE)
    dfmat_line <- dfm(line.toks)
    dfmat_line_matched <- dfm_match(dfmat_line, features = featnames(dfmat_train))
    
    predicted_class.svm <- predict(model, newdata = dfmat_line_matched, probability = TRUE)
    res.sentim <- ifelse(as.numeric(predicted_class.svm) == 1,'Этот комментарий позитивный', 'Этот комментарий негативный')
    
    probab <- attr(predicted_class.svm, "probabilities")
    if(probab[1,1] > probab[1,2]){
      res.probab <- paste("с вероятностью", round(probab[1,1], digits = 2))
    }
    else{
      res.probab <- paste("с вероятностью", round(probab[1,2], digits = 2))
    }
    result <- paste(res.sentim, res.probab, sep = " ")
    print(result)
    
  })
  
  # Status/Output Text Box
  output$contents <- renderPrint({
    if (input$submitbutton > 0) { 
#      if(input$sentence == "")
#     {
#        isolate("Расчет не выполнен") 
#      }
#      else {
        isolate("Расчет выполнен") 
#      }
    } else {
      return("Сервер готов к работе")
    }
  })
  

  # Вывод предложения
  output$textout <- renderText({
    if (input$submitbutton > 0) { 
#      if(input$sentence == ""){
#        isolate("Ошибка. Не введено предложение для анализа") 
#     }
#      else{
        isolate(sentenceInput()) 
#      }
         } 
    })
  output$wordcloud <- renderWordcloud2({
    set.seed(222)
    df <- as.data.frame(com)
    corpus <- iconv(df$com, to = "UTF-8")
    corpus <- Corpus(VectorSource(corpus))
    corpus <- tm_map(corpus, tolower)
    corpus <- tm_map(corpus, removePunctuation)
    corpus <- tm_map(corpus, removeNumbers)
    cleanset <- tm_map(corpus, removeWords, stopwords('russian'))
    cleanset <- tm_map(cleanset, removeWords, c('это'))
    dtm <- TermDocumentMatrix(cleanset)
    dtm <- as.matrix(dtm)
    w <- rowSums(dtm)
    w <- subset(w, w >= 4)
    w <- data.frame(names(w), w)
    colnames(w) <- c('word', 'freq')
    wc.col <- c("#000000", "#FE654F", "#95C623", "#8ACDEA")
    wordcloud2(w,
               size = 0.9,
               color = rep_len(wc.col,nrow(w)),
               fontFamily = "Google Sans",
               shape = 'circle',
               #widgetsize = 1.5,
               rotateRatio = 0.2,
               )
    
  })
  output$plot <- renderPlot({
    beta.svm <- drop(t(model$coefs) %*% dfmat_train[model$index,])
    plot(colSums(dfmat_train),
         beta.svm,
         pch = 19,
         col = rgb(0,0,0,.3),
         cex =.5, 
         log = "x", 
         main = "Коэффициенты SVM-модели",
         ylab = "<--- Негативные комм. --- Позитивные комм.--->", 
         xlab = "Общее количество появлений", 
         xlim = c(1,200)
         )
    text(colSums(dfmat_train),beta.svm, colnames(dfmat_train), pos = 4, cex = .9)
    
  }, height = 'auto')
}

####################################
# Create the shiny app             #
####################################
shinyApp(ui = ui, server = server)
