# 🐾 Yapay Zeka Destekli Hayvan Görüntü Sınıflandırıcı

Bu proje, ResNet50 derin öğrenme mimarisi kullanarak hayvan fotoğraflarını sınıflandıran bir yapay zeka uygulamasıdır. Sistem, eğitim sonrası kullanıcı dostu bir web arayüzü ile görüntüleri sınıflandırabilir ve sonuçları kullanıcıya gösterir.

## 📋 Proje Özellikleri

- **ResNet50** mimarisi ile transfer learning yaklaşımı
- Görüntü ön işleme teknikleri (boyutlandırma, kırpma, veri artırma)
- Eğitim performansını görselleştirme araçları
- Streamlit ile kullanıcı dostu web arayüzü
- Detaylı performans metrikleri (accuracy, precision, recall, F1-score)
- Kamera veya dosya yükleme desteği

## 🚀 Kurulum

Projeyi çalıştırmak için aşağıdaki adımları izleyin:

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install torch torchvision matplotlib numpy scikit-learn seaborn streamlit pillow
```

2. Veri seti klasör yapısını oluşturun:
```
Data/
├── train/
│   ├── kedi/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── kopek/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── kedi/
    │   ├── image1.jpg
    │   └── ...
    ├── kopek/
    │   ├── image1.jpg
    │   └── ...
    └── ...
```

## 🔄 Modeli Eğitme

Modeli eğitmek için aşağıdaki komutu çalıştırın:

```bash
python train.py
```

Eğitim işlemi sonunda aşağıdaki dosyalar oluşturulacaktır:
- `animal_classifier_resnet50.pth`: Eğitilmiş model dosyası
- `class_names.txt`: Sınıf isimleri listesi
- `training_history.png`: Eğitim grafiği (loss ve accuracy)
- `confusion_matrix.png`: Karışıklık matrisi
- `class_metrics.png`: Sınıf bazlı metrikler grafiği
- `model_metrics.txt`: Detaylı performans metrikleri
- `sample_images.png`: Örnek eğitim görselleri

## 🖥️ Web Arayüzünü Çalıştırma

Streamlit web arayüzünü başlatmak için aşağıdaki komutu çalıştırın:

```bash
streamlit run app.py
```

Bu komut tarayıcınızda web arayüzünü açacaktır (genellikle http://localhost:8501).

## 📊 Model Performansı

Eğitim sürecinde model performansı aşağıdaki metrikler kullanılarak ölçülmüştür:
- **Accuracy (Doğruluk)**: Doğru tahmin edilen görüntülerin tüm görüntülere oranı
- **Precision (Kesinlik)**: Pozitif olarak tahmin edilen örneklerin gerçekten pozitif olma oranı
- **Recall (Geri çağırma)**: Gerçek pozitif örneklerin ne kadarının pozitif olarak tespit edildiği
- **F1-Score**: Precision ve Recall'ın harmonik ortalaması

## 🧰 Proje Yapısı

```
├── train.py               # Model eğitim kodu
├── app.py                 # Streamlit web arayüzü kodu
├── Data/                  # Veri seti klasörü
│   ├── train/             # Eğitim verileri
│   └── val/               # Doğrulama verileri
├── animal_classifier_resnet50.pth    # Eğitilmiş model (eğitim sonrası oluşur)
├── class_names.txt        # Sınıf isimleri (eğitim sonrası oluşur)
├── training_history.png   # Eğitim grafikleri (eğitim sonrası oluşur) 
├── confusion_matrix.png   # Karışıklık matrisi (eğitim sonrası oluşur)
├── class_metrics.png      # Sınıf bazlı metrikler (eğitim sonrası oluşur)
├── model_metrics.txt      # Performans metrikleri (eğitim sonrası oluşur)
├── sample_images.png      # Örnek görüntüler (eğitim sonrası oluşur)
└── README.md              # Proje dokümantasyonu
```

## 🔍 Kod Açıklamaları

### train.py
- **Veri Yükleme ve Ön İşleme**:
  - Görüntüleri normalleştirme, boyutlandırma
  - Veri artırma teknikleri (rastgele kırpma, döndürme, renk ayarlamaları)
  - Eğitim ve validasyon verilerini ayırma

- **Model Eğitimi**:
  - ResNet50 mimarisini yükleme ve son katmanını yeniden yapılandırma
  - Transfer learning uygulaması (ImageNet ile önceden eğitilmiş)
  - Kayıp fonksiyonu ve optimizer ayarları
  - Öğrenme hızı planlaması

- **Performans Değerlendirme**:
  - Eğitim ve validasyon kayıp/doğruluk grafikleri
  - Confusion matrix hesaplama ve görselleştirme
  - Precision, recall ve F1 skoru hesaplama
  - Sınıf bazlı performans analizi

- **Model Kaydetme**:
  - Eğitilmiş model ağırlıklarını kaydetme
  - Sınıf isimlerini kaydetme

### app.py
- **Web Arayüzü**:
  - Streamlit ile kullanıcı dostu arayüz
  - Görüntü yükleme ve kamera kullanım seçenekleri
  - Tahmin ve olasılık gösterimi

- **Model Kullanımı**:
  - Eğitilmiş modeli yükleme
  - Görüntüleri ön işleme
  - Tahmin yapma ve sınıflandırma sonuçlarını hesaplama

## 📱 Arayüz Kullanımı

1. Streamlit arayüzü başlatın: `streamlit run app.py`
2. "Dosya Yükle" veya "Kamera Kullan" seçeneğini seçin
3. Bir görüntü yükleyin veya fotoğraf çekin
4. "Tahmin Et" butonuna tıklayın
5. Model tahminini ve olasılık dağılımını görüntüleyin

## 🛠️ Teknik Detaylar

- **Model**: ResNet50 (transfer learning)
- **Görüntü boyutu**: 224x224 piksel
- **Normalizasyon**: ImageNet standartları (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Optimizer**: SGD (Stochastic Gradient Descent)
- **Learning rate**: 0.001 (her 7 adımda 0.1 ile çarpılarak azaltılır)
- **Epochs**: 25
- **Batch size**: 32
- **Loss function**: CrossEntropyLoss

## 📋 Gereksinimler

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- scikit-learn
- seaborn
- Streamlit
- Pillow

## 📋 Proje Sonuçları

Eğitim tamamlandıktan sonra aşağıdaki önemli bilgilere erişilebilir:

1. **Eğitim Performansı**: `training_history.png` dosyası, eğitim ve doğrulama sürecindeki loss ve accuracy değerlerinin değişimini gösteren grafikleri içerir.

2. **Confusion Matrix**: `confusion_matrix.png` dosyası, modelin farklı hayvan sınıflarındaki başarısını ve hata dağılımını gösterir.

3. **Sınıf Bazlı Metrikler**: `class_metrics.png` dosyası, her bir hayvan sınıfı için precision, recall ve F1 skorlarını görsel olarak sunar.

4. **Genel Performans Metrikleri**: `model_metrics.txt` dosyası, modelin genel performansını ve sınıf bazlı detaylı metrikler içerir.

## 📝 Lisans

Bu proje açık kaynaklıdır, eğitim ve geliştirme amaçlı olarak özgürce kullanabilirsiniz.

## 👨‍💻 İletişim

Herhangi bir soru veya geri bildirim için iletişime geçebilirsiniz.
