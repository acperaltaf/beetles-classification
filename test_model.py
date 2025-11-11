"""
Script simple para probar el modelo de clasificación de escarabajos.

Uso:
    python test_model.py
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================================
# CONFIGURACIÓN - MODIFICA ESTAS RUTAS SEGÚN TU CASO
# ============================================================================

MODEL_PATH = 'model_weights/beetle_classifier_resnet50_train_aug_val_real_test_4.pth'  # Ruta a tu modelo .pth
TEST_DATA_DIR = 'images_for_testing/test_this'  # Directorio con imágenes organizadas por carpetas (una carpeta por clase)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 1. CLASE DATASET
# ============================================================================

class BeetleDataset(Dataset):
    """Dataset simple para cargar imágenes de escarabajos."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ============================================================================
# 2. CARGAR DATOS
# ============================================================================

def load_test_images(data_dir):
    """
    Carga TODAS las imágenes de test desde un directorio organizado por clases.
    
    Estructura esperada:
        data_dir/
            clase1/
                img1.jpg
                img2.jpg
            clase2/
                img3.jpg
            ...
    
    Returns:
        image_paths: Lista de rutas a imágenes
        labels: Lista de etiquetas (índices numéricos)
        class_names: Lista con nombres de las clases
    """
    print(f"📂 Cargando imágenes desde: {data_dir}")
    
    image_paths = []
    labels = []
    class_names = []
    
    # Recorrer carpetas (cada carpeta = una clase)
    for class_idx, class_folder in enumerate(sorted(os.listdir(data_dir))):
        class_path = os.path.join(data_dir, class_folder)
        
        if os.path.isdir(class_path):
            class_names.append(class_folder)
            
            # Obtener todas las imágenes de esta clase
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_path, img_file))
                    labels.append(class_idx)
    
    print(f"✅ {len(class_names)} clases encontradas")
    print(f"✅ {len(image_paths)} imágenes de test")
    
    return image_paths, labels, class_names

# ============================================================================
# 3. CARGAR MODELO
# ============================================================================

def load_model(model_path, device):
    """
    Carga el modelo desde un archivo .pth.
    Detecta automáticamente el número de clases.
    """
    print(f"\n📥 Cargando modelo desde: {model_path}")
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Obtener state_dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            saved_class_names = checkpoint.get('class_names', None)
        else:
            state_dict = checkpoint
            saved_class_names = None
    else:
        state_dict = checkpoint
        saved_class_names = None
    
    # Detectar número de clases desde la última capa
    if 'fc.3.weight' in state_dict:
        num_classes = state_dict['fc.3.weight'].shape[0]
    elif 'fc.weight' in state_dict:
        num_classes = state_dict['fc.weight'].shape[0]
    else:
        raise ValueError("No se pudo detectar el número de clases del modelo")
    
    print(f"✅ Número de clases detectado: {num_classes}")
    if saved_class_names:
        print(f"✅ Clases en el modelo: {saved_class_names}")
    
    # Crear arquitectura ResNet50
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # Cargar pesos
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modelo cargado en {device}")
    
    return model, num_classes, saved_class_names

# ============================================================================
# 4. EVALUAR MODELO
# ============================================================================

def evaluate_model(model, test_loader, class_names, device):
    """Evalúa el modelo y muestra métricas."""
    
    print("\n🔄 Evaluando modelo...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calcular accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\n{'='*80}")
    print(f"📊 RESULTADOS")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Identificar qué clases están presentes en los datos de test
    unique_labels = np.unique(all_labels)
    present_class_names = [class_names[i] for i in unique_labels]
    
    print(f"\n📋 Reporte por clase:")
    print(f"{'='*80}")
    print(f"ℹ️  Clases presentes en test: {len(unique_labels)}/{len(class_names)}")
    print(f"   {present_class_names}")
    print(f"{'='*80}\n")
    
    # Usar labels parameter para especificar solo las clases presentes
    print(classification_report(
        all_labels, all_preds, 
        labels=unique_labels,
        target_names=present_class_names, 
        digits=4,
        zero_division=0
    ))
    
    return all_preds, all_labels, accuracy

# ============================================================================
# 5. PREDECIR IMAGEN INDIVIDUAL
# ============================================================================

def predict_image(model, image_path, class_names, device, transform):
    """
    Predice la clase de una imagen individual.
    
    Args:
        model: Modelo cargado
        image_path: Ruta a la imagen
        class_names: Lista de nombres de clases
        device: Dispositivo (cuda/cpu)
        transform: Transformaciones a aplicar
    
    Returns:
        predicted_class: Nombre de la clase predicha
        confidence: Confianza de la predicción (0-1)
        top_3: Lista de tuplas (clase, probabilidad) con top 3
    """
    # Cargar y transformar imagen
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predecir
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, k=min(3, len(class_names)))
    
    # Convertir a listas
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    
    predicted_class = class_names[top_indices[0]]
    confidence = top_probs[0]
    
    top_3 = [(class_names[idx], prob) for idx, prob in zip(top_indices, top_probs)]
    
    return predicted_class, confidence, top_3

# ============================================================================
# 6. MAIN - SCRIPT PRINCIPAL
# ============================================================================

def main():
    """Función principal."""
    
    print("="*80)
    print("🧪 TESTING DE MODELO - CLASIFICACIÓN DE ESCARABAJOS")
    print("="*80)
    
    # Transformaciones (las mismas que en entrenamiento)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 1. Cargar modelo
    model, num_classes, saved_class_names = load_model(MODEL_PATH, DEVICE)
    
    # 2. Cargar datos de test
    test_paths, test_labels, class_names = load_test_images(TEST_DATA_DIR, test_split=0.2)
    
    # Verificar compatibilidad
    if len(class_names) != num_classes:
        print(f"\n⚠️  ADVERTENCIA:")
        print(f"   El modelo tiene {num_classes} clases")
        print(f"   Los datos de test tienen {len(class_names)} clases")
        print(f"   Esto puede causar errores o resultados incorrectos.")
        
        respuesta = input("\n¿Deseas continuar de todos modos? (s/n): ")
        if respuesta.lower() != 's':
            print("❌ Prueba cancelada")
            return
    
    # 3. Crear DataLoader
    test_dataset = BeetleDataset(test_paths, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Evaluar modelo
    all_preds, all_labels, accuracy = evaluate_model(model, test_loader, class_names, DEVICE)
    
    # 5. Ejemplo de predicción individual
    print(f"\n{'='*80}")
    print(f"🔍 EJEMPLO: Predicción en imagen individual")
    print(f"{'='*80}")
    
    random_idx = np.random.randint(0, len(test_paths))
    test_image = test_paths[random_idx]
    true_label = class_names[test_labels[random_idx]]
    
    predicted_class, confidence, top_3 = predict_image(
        model, test_image, class_names, DEVICE, transform
    )
    
    print(f"\n📸 Imagen: {os.path.basename(test_image)}")
    print(f"✅ Clase real: {true_label}")
    print(f"🎯 Predicción: {predicted_class} ({confidence*100:.2f}%)")
    print(f"\n📊 Top 3 predicciones:")
    for i, (cls, prob) in enumerate(top_3, 1):
        print(f"   {i}. {cls:35s} {prob*100:6.2f}%")
    
    if predicted_class == true_label:
        print(f"\n✅ ¡Predicción CORRECTA!")
    else:
        print(f"\n❌ Predicción INCORRECTA")
    
    print(f"\n{'='*80}")
    print(f"✅ Testing completado")
    print(f"{'='*80}")

# ============================================================================
# EJECUTAR
# ============================================================================

if __name__ == "__main__":
    main()
