# Ridge-Regression-for-Sales-Prediction-Model-Evaluation-and-Hyperparameter-Tuning

![image](https://github.com/user-attachments/assets/4f462fd4-ef61-4797-baa5-8686adaff416)


# Ridge Regression with Polynomial Features

Bu proje, Ridge regresyonu ve polinom özellikleri kullanarak bir model oluşturmayı ve değerlendirmeyi amaçlamaktadır. Model, belirli bir veri setine uygulandıktan sonra, doğrusal regresyonun yanı sıra polinom özellikleri ile modelin performansı iyileştirilmeye çalışılmıştır.

## Proje Amacı

Bu projenin amacı, Ridge regresyonu kullanarak veri setindeki özelliklere dayalı tahminlerde bulunmak ve modelin doğruluğunu farklı regularizasyon parametreleri ve polinom özellikleri ile optimize etmektir. Ayrıca, modelin performansını değerlendirmek ve görselleştirmek için çeşitli metrikler kullanılmıştır.

## Kullanılan Yöntemler

- **Ridge Regresyonu**: Ridge regresyonu, verinin doğrusal ilişkisini modellemek için kullanılmıştır. Regularizasyon parametresi (alpha) ile overfitting (aşırı uyum) engellenmeye çalışılmıştır.
- **Polinom Özellikler**: Veriye polinom özellikler (derece 2) eklenerek modelin doğrusal olmayan ilişkileri öğrenebilmesi sağlanmıştır.
- **Grid Search**: Modelin regularizasyon gücünü optimize etmek için grid search yöntemi kullanılmıştır.
- **Cross-validation**: Modelin genellenebilirliğini test etmek için çapraz doğrulama kullanılmıştır.

## Kullanılan Kütüphaneler

- `pandas`: Veri manipülasyonu ve analizi.
- `numpy`: Sayısal hesaplamalar.
- `scikit-learn`: Makine öğrenmesi algoritmaları ve model değerlendirme.
- `matplotlib`: Görselleştirme ve grafikler.
- `seaborn`: Gelişmiş görselleştirme.

## Kurulum

Projenin gereksinimlerini yüklemek için aşağıdaki komutları kullanabilirsiniz:

1. Bu projeyi klonlayın:
   ```bash
   git clone https://github.com/deliprofesor/Ridge-Regression-for-Sales-Prediction-Model-Evaluation-and-Hyperparameter-Tuning.git
   cd Ridge-Regression-for-Sales-Prediction-Model-Evaluation-and-Hyperparameter-Tuning

pip install -r requirements.txt

# Model Eğitimi ve Değerlendirme

## İlk Ridge Regresyonu:

Ridge regresyonu modelini, standart ölçeklendirme ve regularizasyon (cezalandırma) ile bir pipeline içinde eğittiniz.
- **MSE (Ortalama Kare Hatası): 58.23**
- **RMSE (Karekök Ortalama Kare Hatası): 7.63**
- **R² skoru: 0.7753** (bu, modelin verideki varyansın yaklaşık %77.53'ini açıkladığını gösteriyor).

## Görselleştirme:

**Gerçek ve Tahmin Edilen Satışlar:** Gerçek satışlarla tahmin edilen satışlar arasındaki ilişkiyi gösteren bir dağılım grafiği çizildi.
**Regresyon Doğrusu:** Grafikte kırmızı çizgi, gerçek ve tahmin edilen değerler arasındaki doğrusal ilişkiyi gösteriyor.
**Kalıntılar Dağılımı:** Kalıntıların histogramı, iyi bir model performansı için genellikle normal dağılım göstermelidir.

## Model Optimizasyonu ve Grid Search

En İyi Alpha (Regularizasyon Gücü): Grid search ile en iyi alpha değeri 0.1 olarak bulundu.
Grid Search Sonuçları: Farklı regularizasyon parametreleri (alpha) denendi ve 0.1 değeri en iyi performansı gösterdi.

## Polinomsal Özellikler ve Çapraz Doğrulama

**Polinomsal Regresyon:** İkinci derece polinomsal özellikler eklediniz ve çapraz doğrulama ile modeli değerlendirdiniz.
**Çapraz doğrulama MSE skorları:** -419.72 ile -58.16 arasında değişti.
**Ortalama Çapraz Doğrulama MSE:** -271.73 (negatif ortalama MSE, hata teriminin minimize edilmesi beklendiği için normaldir).

## Final Model Değerlendirmesi

**Polinomsal Özelliklerle R² Skoru:**

**Eğitim verisi:** 0.9462, bu, modelin eğitim verisinde çok iyi bir uyum sağladığını gösteriyor.
**Test verisi:** 0.7323, bu da modelin eğitim verisiyle iyi uyum sağlarken, test verisi üzerinde biraz daha düşük bir doğruluk sergilediğini gösteriyor.

## Genel Değerlendirme ve Sonuçlar

Ridge regresyonu modeliniz test setinde %77.53 kadar varyansı açıklayarak gayet iyi bir performans gösterdi.
Polinomsal özellikler ekleyerek modelin varyansı açıklama kabiliyetini önemli ölçüde artırdınız (eğitim verisindeki yüksek R² skoru bunu gösteriyor).
Ridge regularizasyonu (alpha=0.1), aşırı öğrenmeyi (overfitting) engellemeye yardımcı oldu, çünkü test seti üzerinde de hala iyi bir performans sergiledi.

