#instalamos librerias#
install.packages("devtools")
devtools::install_github("rstudio/keras")
library(keras)
install_keras()
install.packages("e1071")
install.packages("readxl")
source("http://bioconductor.org/biocLite.R")
biocLite("switchBox")
#cargamos las librerias#
require(switchBox)
library(keras)
library(e1071)
#cargamos pesos de la red VGG16#
load("F:/UCC/variablescurso.Rdata")
load("/variablescurso.Rdata")
# base_model <- application_vgg16(weights = 'imagenet') cargar pesos desde la red #
base_model<-unserialize_model(vgg16) #reconvertir modelo#
#vemos el modelo#
base_model
#cargamos una imagen de ejemplo, y la dimencionamos al tamaño de entrada de la red#
img_path <- "/fundussick.jpg"
img_path <- "F:/UCC/fundus.jpg"
img <- image_load(img_path,target_size = c(224,224))
x <- image_to_array(img)
x <- array_reshape(x, c(1, dim(x)))
x <- imagenet_preprocess_input(x)
#corremos la imagen y vemos la prediccion de la red#
preds <- base_model %>% predict(x)
imagenet_decode_predictions(preds, top = 3)[[1]]
#extraemos caracteristicas de la primera capa fully connected#
model <- keras_model(inputs = base_model$input, 
                     outputs = get_layer(base_model, 'fc1')$output)
fc1_features <- model %>% predict(x)
#exploramos las variables
fc1_plot<- as.numeric(fc1_features)
boxplot(fc1_plot)
#optimizamos un modelo de SVM#
tunedsvm0<-best.tune(svm,value,labels,probability = TRUE)
#clasificamos la imagen ejemplo#
prediccion0<-predict(tunedsvm0,fc1_features,probability = TRUE)
#creamos un modelo de k-tsp#
value<-t(value)
fc1_features<-t(fc1_features)
rownames(fc1_features)<-rownames(value)
classifiercomp<- SWAP.Train.KTSP(value,labels, krange=2:25, FilterFunc = SWAP.Filter.Wilcoxon,verbose = TRUE)
#clasificamos la imagen ejempo#
trainingPrediction <- SWAP.KTSP.Classify(fc1_features, classifiercomp)
estadistica<-SWAP.KTSP.Statistics(fc1_features, classifiercomp)

