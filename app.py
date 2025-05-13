#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# ÖNEMLİ = streamlit run app.py yaparak terminalden çalıştırınız

Hayvan Görüntü Sınıflandırıcı - Web Arayüzü
Streamlit kullanarak eğitilmiş model ile hayvan görüntülerini sınıflandıran web arayüzü
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import PIL
from PIL import Image
import numpy as np
import io
import os

class AnimalClassifierApp:
    """
    Eğitilmiş ResNet50 modeli ile hayvan sınıflandırıcı web uygulaması
    """
    
    def __init__(self, model_path='animal_classifier_resnet50.pth'):
        """
        Parametreler:
            model_path (str): Eğitilmiş model dosyasının yolu
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        try:
            # Model yoksa uyarı ver
            if not os.path.exists(self.model_path):
                st.error(f"Model dosyası bulunamadı: {self.model_path}")
                st.info("Lütfen önce 'train.py' dosyasını çalıştırarak modeli eğitin.")
                return False
            
            # Modeli yükle
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # ResNet50 modelini oluştur
            self.model = models.resnet50(weights=None)
            
            # Sınıf sayısına göre son katmanı ayarla
            self.class_names = checkpoint['class_names']
            num_classes = len(self.class_names)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
            
            # Model ağırlıklarını yükle
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            return True
        
        except Exception as e:
            st.error(f"Model yüklenirken hata oluştu: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """
        Yüklenen görüntüyü model için ön işleme tabi tut
        
        Parametreler:
            image (PIL.Image): İşlenecek görüntü
            
        Dönüş:
            torch.Tensor: İşlenmiş ve normalize edilmiş görüntü tensörü
        """
        # Görüntüyü RGB'ye dönüştür (eğer değilse)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Dönüşümleri uygula
        image_tensor = self.transform(image)
        
        # Batch boyutunu ekle
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image):
        """
        Görüntüyü sınıflandır
        
        Parametreler:
            image (PIL.Image): Sınıflandırılacak görüntü
            
        Dönüş:
            tuple: (tahmin edilen sınıf, olasılık yüzdesi)
        """
        # Görüntüyü ön işle
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Tahmin yap
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # En yüksek olasılığa sahip sınıfı bul
            top_prob, top_class = torch.max(probabilities, 0)
            
            # Tüm olasılıkları listeye çevir
            all_probs = probabilities.cpu().numpy()
            
            return top_class.item(), top_prob.item(), all_probs
    
    def run(self):
        """Streamlit uygulamasını çalıştır"""
        st.set_page_config(
            page_title="Hayvan Görüntü Sınıflandırıcı",
            page_icon="🐾",
            layout="wide"
        )
        
        # Başlık ve açıklama
        st.title("🐾 Hayvan Görüntü Sınıflandırıcı")
        st.markdown("""
        Bu uygulama, yüklediğiniz hayvan görüntülerini otomatik olarak sınıflandırır.
        ResNet50 mimarisi kullanılarak eğitilmiş bir yapay zeka modeli kullanılmıştır.
        """)
        
        # Modeli yükle
        model_loaded = self.load_model()
        
        if not model_loaded:
            return
        
        # Yan menü - Sınıf bilgileri
        with st.sidebar:
            st.header("Model Bilgileri")
            st.write(f"Tanınabilen hayvan türleri ({len(self.class_names)}):")
            for class_name in self.class_names:
                st.write(f"• {class_name}")
            
            st.markdown("---")
            st.write("Yapay Zeka Modeli: ResNet50")
            st.write(f"Çalıştığı Cihaz: {self.device}")
        
        # Ana içerik - Görüntü yükleme
        st.header("Görüntü Yükle")
        
        # Kullanıcıdan görüntü yükleme seçenekleri
        upload_option = st.radio(
            "Görüntü yükleme yöntemi seçin:",
            ["Dosya Yükle", "Kamera Kullan"]
        )
        
        # Tahmin sonuçlarını tutacak değişkenler
        prediction_class = None
        prediction_prob = None
        all_probs = None
        image = None
        
        # Dosya yükleme seçeneği
        if upload_option == "Dosya Yükle":
            uploaded_file = st.file_uploader("Bir hayvan görüntüsü yükleyin", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Yüklenen dosyayı PIL Image'e dönüştür
                image = Image.open(uploaded_file)
                
                # Görüntüyü göster
                st.image(image, caption="Yüklenen Görüntü", use_column_width=True)
                
                # Tahmin butonuna basıldığında
                if st.button("Tahmin Et"):
                    with st.spinner("Görüntü analiz ediliyor..."):
                        # Tahmin yap
                        prediction_class, prediction_prob, all_probs = self.predict(image)
        
        # Kamera seçeneği
        else:
            camera_image = st.camera_input("Kamera ile Fotoğraf Çekin")
            
            if camera_image is not None:
                # Kameradan alınan görüntüyü PIL Image'e dönüştür
                image = Image.open(camera_image)
                
                # Tahmin butonuna basıldığında
                if st.button("Tahmin Et"):
                    with st.spinner("Görüntü analiz ediliyor..."):
                        # Tahmin yap
                        prediction_class, prediction_prob, all_probs = self.predict(image)
        
        # Tahmin sonuçlarını göster
        if prediction_class is not None:
            st.header("Tahmin Sonucu")
            
            # Tahmin edilen sınıf ve olasılık
            st.success(f"Görüntüdeki hayvan: **{self.class_names[prediction_class]}**")
            st.progress(prediction_prob)
            st.write(f"Emin olma düzeyi: **%{prediction_prob*100:.2f}**")
            
            # Tüm sınıfların olasılıklarını göster
            st.subheader("Tüm Sınıfların Olasılıkları")
            
            # Olasılıkları sırala
            sorted_indices = np.argsort(all_probs)[::-1]
            
            # En yüksek 5 tahmin
            col1, col2 = st.columns(2)
            
            with col1:
                for i in range(min(5, len(self.class_names))):
                    idx = sorted_indices[i]
                    st.write(f"{self.class_names[idx]}: **%{all_probs[idx]*100:.2f}**")
            
            # Olasılık çubuklarını göster
            with col2:
                # Olasılık grafiği
                prob_data = {
                    'Sınıf': [self.class_names[sorted_indices[i]] for i in range(min(5, len(self.class_names)))],
                    'Olasılık': [all_probs[sorted_indices[i]] * 100 for i in range(min(5, len(self.class_names)))]
                }
                
                for i in range(min(5, len(self.class_names))):
                    st.progress(all_probs[sorted_indices[i]])


# Ana uygulama
def main():
    app = AnimalClassifierApp()
    app.run()

if __name__ == "__main__":
    main()