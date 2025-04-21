
---
title: "Prediksi Klasifikasi Latihan Angkat Beban Menggunakan Data Akselerometer"
author: "User"
output: html_document
---

# Pendahuluan

Proyek ini bertujuan untuk memprediksi cara melakukan latihan angkat beban menggunakan data akselerometer. Variabel target yang digunakan adalah `classe`, yang mewakili kategori atau kelas dari latihan yang dilakukan. Data akselerometer mencakup berbagai fitur yang diambil selama latihan angkat beban.

# Persiapan Data

Pertama-tama, kita akan memuat dan menyiapkan data akselerometer untuk analisis.

```{r setup, include=TRUE}
library(tidyverse)
library(caret)
library(randomForest)

# Memuat data
data <- read.csv("data_latihan_angkat_beban.csv")
head(data)
```

Pada bagian ini, kita memuat dataset yang berisi data latihan angkat beban yang berasal dari sensor akselerometer. Dataset ini memiliki beberapa fitur yang berkaitan dengan gerakan tubuh selama latihan.

# Eksplorasi Data

Sebelum membangun model prediksi, mari kita eksplorasi data untuk memahami pola dan distribusi data.

```{r}
# Melihat distribusi target 'classe'
ggplot(data, aes(x = classe)) +
  geom_bar() +
  labs(title = "Distribusi Kelas Latihan Angkat Beban")
```

Dari grafik di atas, kita dapat melihat distribusi kelas latihan angkat beban. Ini penting untuk memastikan bahwa data tidak terlalu imbalanced, yang bisa mempengaruhi performa model.

# Pembagian Data Latih dan Uji

Selanjutnya, kita akan membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk melatih model, dan data uji akan digunakan untuk evaluasi model.

```{r}
# Pembagian data menjadi data latih dan uji
set.seed(123)
trainIndex <- createDataPartition(data$classe, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Menampilkan ukuran data latih dan uji
dim(trainData)
dim(testData)
```

Pada langkah ini, data dibagi dengan rasio 70% untuk pelatihan dan 30% untuk pengujian.

# Penjelasan Algoritma

Random Forest adalah algoritma pembelajaran ensemble berbasis pohon keputusan. Ia bekerja dengan membangun banyak pohon keputusan pada subset acak dari data pelatihan dan kemudian menggabungkan prediksi dari masing-masing pohon untuk menentukan kelas akhir. Keunggulan Random Forest terletak pada:

- Kemampuannya mengatasi overfitting dengan menggabungkan banyak pohon,
- Tahan terhadap noise dan data hilang,
- Memberikan estimasi pentingnya fitur.

# Model Prediksi

Untuk model prediksi, kita akan menggunakan algoritma Random Forest, yang sering digunakan dalam klasifikasi.

```{r}
# Membangun model Random Forest
model_rf <- randomForest(classe ~ ., data = trainData, ntree = 100)

# Menampilkan hasil model
print(model_rf)
```

# Evaluasi Model

Setelah membangun model, mari kita evaluasi kinerjanya menggunakan data uji.

```{r}
# Memprediksi kelas dengan data uji
predictions <- predict(model_rf, newdata = testData)

# Menghitung matriks kebingungannya
confusionMatrix(predictions, testData$classe)
```

Matriks kebingungannya menunjukkan seberapa baik model memprediksi kelas pada data uji, memberikan gambaran tentang akurasi dan performa model dalam klasifikasi.

# Validasi Silang dan Estimasi Kesalahan

Untuk memperkirakan kesalahan di luar sampel secara lebih akurat, kita menggunakan validasi silang.

```{r}
# Menggunakan 5-fold cross-validation
set.seed(123)
control <- trainControl(method = "cv", number = 5)

# Melatih model dengan validasi silang
model_cv <- train(classe ~ ., data = trainData, method = "rf",
                  trControl = control, ntree = 100)

# Hasil validasi silang
print(model_cv)

# Akurasi rata-rata dari validasi silang
mean(model_cv$resample$Accuracy)

# Menghitung kesalahan di luar sampel
oos_error <- 1 - mean(model_cv$resample$Accuracy)
oos_error
```

# Kesimpulan

Model Random Forest yang dibangun menunjukkan kemampuan yang baik dalam memprediksi kelas latihan angkat beban berdasarkan data akselerometer. Dengan validasi silang, diperoleh estimasi kesalahan di luar sampel yang rendah, yang menunjukkan kemampuan model dalam melakukan generalisasi terhadap data baru.
