#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hayvan Görüntü Sınıflandırıcı - Eğitim Modülü
ResNet50 modeli kullanarak hayvan görüntüleri tanıyan bir model eğitir
"""

import os
import time 
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

class ModelTrainer:
    """
    Hayvan görüntü sınıflandırıcı modelinin eğitimini yönetir.
    ResNet50 modeli kullanarak bir transfer learning uygulaması yapar.
    """
    
    def __init__(self, data_dir='Data', batch_size=32, num_epochs=25, learning_rate=0.001):
        """
        Parametreler:
            data_dir (str): Veri klasörünün yolu
            batch_size (int): Eğitim batch boyutu
            num_epochs (int): Eğitim dönemi sayısı
            learning_rate (float): Öğrenme hızı
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = []
        self.dataloaders = {}
        self.dataset_sizes = {}
        
        print(f"Cihaz kullanımı: {self.device}")
        
    def prepare_data(self):
        """Veri ön işleme ve yükleme işlemlerini gerçekleştirir."""
        
        # Veri artırma ve normalleştirme için dönüşümler
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Veri setlerini yükle
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        
        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                     batch_size=self.batch_size,
                                                     shuffle=True, 
                                                     num_workers=4)
                      for x in ['train', 'val']}
        
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        self.class_names = image_datasets['train'].classes
        print(f"Sınıf isimleri: {self.class_names}")
        print(f"Eğitim veri seti boyutu: {self.dataset_sizes['train']}")
        print(f"Doğrulama veri seti boyutu: {self.dataset_sizes['val']}")
        
        # Örnek resimler göster
        self.visualize_sample_images()
    
    def visualize_sample_images(self, num_images=4):
        """Veri setinden örnek görüntüleri görüntüler."""
        
        # Örnek resimleri al
        images, labels = next(iter(self.dataloaders['train']))
        images = images[:num_images]
        labels = labels[:num_images]
        
        # Görüntüler için normalleştirmeyi geri al
        img_np = images.numpy().transpose((0, 2, 3, 1))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Örnek görüntüleri göster
        plt.figure(figsize=(12, 6))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(img_np[i])
            plt.title(f"{self.class_names[labels[i]]}")
            plt.axis('off')
        
        plt.suptitle("Örnek Eğitim Görüntüleri", fontsize=16)
        plt.savefig('sample_images.png')
        plt.close()
        print("Örnek görüntüler 'sample_images.png' dosyasına kaydedildi.")
    
    def train_model(self):
        """ResNet50 modelini eğitir ve sonuçları kaydeder."""
        # ResNet50 modelini yükle
        model = models.resnet50(weights='IMAGENET1K_V1')
        
        # Son fully connected katmanını yeniden tanımla
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.class_names))
        
        model = model.to(self.device)
        
        # Kayıp fonksiyonu ve optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        
        # Öğrenme hızı zamanlayıcısı
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # En iyi modeli saklayacak değişken
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        # Eğitim süresini takip etmek için
        start_time = time.time()
        
        # Eğitim ilerlemesini izlemek için
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        # Eğitim döngüsü
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)
            
            # Her epoch'ta hem eğitim hem de validasyon
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0.0
                running_corrects = 0
                
                # Batch'lerde ilerleme
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Parametreleri sıfırla
                    optimizer.zero_grad()
                    
                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Eğitim aşamasında backward + optimize
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # İstatistikleri takip et
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                
                # Geçmiş metrikleri kaydet
                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc.item())
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc.item())
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # En iyi modeli kopyala
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            
            print()
        
        # Toplam eğitim süresini göster
        time_elapsed = time.time() - start_time
        print(f'Eğitim tamamlandı - {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'En iyi validasyon doğruluğu: {best_acc:.4f}')
        
        # En iyi model ağırlıklarını yükle
        model.load_state_dict(best_model_wts)
        
        # Eğitim grafiğini çiz
        self.plot_training_history(train_loss_history, val_loss_history, 
                                  train_acc_history, val_acc_history)
        
        # Confusion matrix ve diğer metrikleri hesapla
        self.evaluate_model(model)
        
        # Modeli kaydet
        self.save_model(model)
        
        return model
    
    def plot_training_history(self, train_loss, val_loss, train_acc, val_acc):
        """Eğitim ve validasyon kayıp/doğruluk grafiklerini çizer."""
        plt.figure(figsize=(12, 5))
        
        # Loss grafiği
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Eğitim Kaybı')
        plt.plot(val_loss, label='Validasyon Kaybı')
        plt.xlabel('Epoch')
        plt.ylabel('Kayıp')
        plt.legend()
        plt.title('Eğitim ve Validasyon Kaybı')
        
        # Accuracy grafiği
        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label='Eğitim Doğruluğu')
        plt.plot(val_acc, label='Validasyon Doğruluğu')
        plt.xlabel('Epoch')
        plt.ylabel('Doğruluk')
        plt.legend()
        plt.title('Eğitim ve Validasyon Doğruluğu')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        print("Eğitim sonuçları 'training_history.png' dosyasına kaydedildi.")
    
    def evaluate_model(self, model):
        """
        Modelin performansını değerlendirir: confusion matrix, precision, recall, F1-score
        """
        model.eval()
        
        # Tüm validasyon verisi için tahminleri topla
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.dataloaders['val']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Confusion matrix hesapla ve görselleştir
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("Confusion matrix 'confusion_matrix.png' dosyasına kaydedildi.")
        
        # Precision, recall ve F1 skorlarını hesapla
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f"\nModel Performansı Metrikleri:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Her sınıf için ayrı skorları hesapla
        class_precision = precision_score(all_labels, all_preds, average=None)
        class_recall = recall_score(all_labels, all_preds, average=None)
        class_f1 = f1_score(all_labels, all_preds, average=None)
        
        # Sınıf bazlı metrikleri görselleştir
        plt.figure(figsize=(12, 6))
        x = np.arange(len(self.class_names))
        width = 0.25
        
        plt.bar(x - width, class_precision, width, label='Precision')
        plt.bar(x, class_recall, width, label='Recall')
        plt.bar(x + width, class_f1, width, label='F1')
        
        plt.xlabel('Sınıflar')
        plt.ylabel('Skor')
        plt.title('Sınıf Bazlı Performans Metrikleri')
        plt.xticks(x, self.class_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('class_metrics.png')
        plt.close()
        print("Sınıf bazlı metrikler 'class_metrics.png' dosyasına kaydedildi.")

        # Tüm metrikleri bir dosyaya kaydet
        with open('model_metrics.txt', 'w') as f:
            f.write(f"Model Performansı Metrikleri:\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            
            f.write("Sınıf Bazlı Metrikler:\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {class_precision[i]:.4f}\n")
                f.write(f"  Recall: {class_recall[i]:.4f}\n")
                f.write(f"  F1 Score: {class_f1[i]:.4f}\n")
        
        print("Model metrikleri 'model_metrics.txt' dosyasına kaydedildi.")
    
    def save_model(self, model):
        """
        Eğitilmiş modeli ve sınıf isimlerini kaydeder
        """
        # Modeli kaydet
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': self.class_names
        }, 'animal_classifier_resnet50.pth')
        
        print("Model 'animal_classifier_resnet50.pth' olarak kaydedildi.")
        
        # Sınıf isimlerini ayrı kaydet (Streamlit için)
        with open('class_names.txt', 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        
        print("Sınıf isimleri 'class_names.txt' dosyasına kaydedildi.")


def main():
    """Ana program çalıştırma fonksiyonu"""
    print("Hayvan Görüntü Sınıflandırma Modeli Eğitimi Başlıyor...")
    
    # ModelTrainer sınıfını başlat
    trainer = ModelTrainer(data_dir='Data', batch_size=32, num_epochs=25)
    
    # Verileri hazırla
    trainer.prepare_data()
    
    # Modeli eğit
    model = trainer.train_model()
    
    print("Eğitim tamamlandı!")


if __name__ == "__main__":
    main()