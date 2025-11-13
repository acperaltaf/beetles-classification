import os
import shutil
from pathlib import Path
import re

# Configuración
source_dir = 'cropped_beetles/images'  # Carpeta con todas las imágenes
destination_dir = 'cropped_beetles/images_no_flash'  # Carpeta para imágenes sin flash

def filter_no_flash_images(source_dir, destination_dir, copy_mode=True):
    """
    Filtra y copia/mueve solo las imágenes tomadas sin flash (contienen '_s_' o '_s.' en el nombre).
    
    Args:
        source_dir (str): Directorio de origen con las imágenes
        destination_dir (str): Directorio de destino para imágenes filtradas
        copy_mode (bool): True para copiar, False para mover
    """
    
    # Crear directorio de destino si no existe
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    
    # Contadores
    total_images = 0
    flash_images = 0
    no_flash_images = 0
    copied_images = 0
    
    # Contadores de errores (imágenes que se filtraron incorrectamente antes)
    errors_found = []
    
    print("🔍 Buscando imágenes...")
    print(f"📁 Directorio de origen: {source_dir}")
    print(f"📁 Directorio de destino: {destination_dir}")
    print(f"{'📋 Modo: COPIAR' if copy_mode else '🚚 Modo: MOVER'}\n")
    
    # Patrón más preciso: busca _s_ o _s. (antes de la extensión)
    # Ejemplos válidos: xxx_s_1.JPG, xxx_s.JPG
    # Ejemplos inválidos: xxx_f_1.JPG, subhyalinus, transistmius
    no_flash_pattern = re.compile(r'_s[_\.]', re.IGNORECASE)
    flash_pattern = re.compile(r'_f[_\.]', re.IGNORECASE)
    
    # Recorrer todas las carpetas de géneros
    for genus_folder in sorted(os.listdir(source_dir)):
        genus_source_path = os.path.join(source_dir, genus_folder)
        
        # Verificar si es una carpeta
        if not os.path.isdir(genus_source_path):
            continue
            
        # Crear la misma estructura de carpetas en destino
        genus_dest_path = os.path.join(destination_dir, genus_folder)
        Path(genus_dest_path).mkdir(parents=True, exist_ok=True)
        
        genus_flash = 0
        genus_no_flash = 0
        
        # Procesar imágenes de cada género
        for img_file in os.listdir(genus_source_path):
            # Verificar si es una imagen
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            total_images += 1
            source_path = os.path.join(genus_source_path, img_file)
            
            # Verificar con patrón más preciso
            is_no_flash = no_flash_pattern.search(img_file)
            is_flash = flash_pattern.search(img_file)
            
            # Si tiene _s_ o _s. es sin flash
            if is_no_flash and not is_flash:
                no_flash_images += 1
                genus_no_flash += 1
                
                # Copiar o mover la imagen
                dest_path = os.path.join(genus_dest_path, img_file)
                
                try:
                    if copy_mode:
                        shutil.copy2(source_path, dest_path)
                    else:
                        shutil.move(source_path, dest_path)
                    copied_images += 1
                except Exception as e:
                    print(f"❌ Error procesando {img_file}: {e}")
            
            # Si tiene _f_ o _f. es con flash
            elif is_flash:
                flash_images += 1
                genus_flash += 1
            
            # Si no tiene ninguno de los dos patrones, reportar como advertencia
            else:
                flash_images += 1
                genus_flash += 1
                if total_images <= 10:  # Mostrar solo los primeros como ejemplo
                    print(f"⚠️  Formato desconocido: {img_file}")
        
        # Mostrar resumen por género
        if genus_no_flash > 0 or genus_flash > 0:
            print(f"📂 {genus_folder}:")
            print(f"   ✅ Sin flash (_s_): {genus_no_flash}")
            print(f"   ⚡ Con flash (_f_): {genus_flash}")
    
    # Resumen final
    print("\n" + "="*60)
    print("📊 RESUMEN FINAL")
    print("="*60)
    print(f"🖼️  Total de imágenes encontradas: {total_images}")
    print(f"✅ Imágenes sin flash (_s_): {no_flash_images} ({no_flash_images/total_images*100:.1f}%)")
    print(f"⚡ Imágenes con flash (_f_): {flash_images} ({flash_images/total_images*100:.1f}%)")
    print(f"{'📋 Imágenes copiadas' if copy_mode else '🚚 Imágenes movidas'}: {copied_images}")
    print("="*60)
    
    return {
        'total': total_images,
        'no_flash': no_flash_images,
        'flash': flash_images,
        'processed': copied_images
    }

# IMPORTANTE: Primero eliminar la carpeta anterior si existe para empezar de cero
if os.path.exists('cropped_beetles/images_no_flash'):
    print("🗑️  Eliminando carpeta anterior 'images_no_flash' para empezar de cero...")
    shutil.rmtree('cropped_beetles/images_no_flash')
    print("✅ Carpeta eliminada\n")

# Ejecutar el filtrado de imágenes sin flash
print("🚀 Iniciando filtrado de imágenes sin flash...\n")
stats = filter_no_flash_images(
    source_dir='cropped_beetles/images',
    destination_dir='cropped_beetles/images_no_flash',
    copy_mode=True  # Cambiar a False si quieres MOVER en lugar de copiar
)

# Verificar resultado
print(f"\n✅ Proceso completado exitosamente!")
print(f"📁 Las imágenes sin flash están en: cropped_beetles/images_no_flash")

# Verificación adicional: buscar ejemplos de cada tipo
print("\n" + "="*60)
print("🔍 VERIFICACIÓN - Ejemplos de archivos procesados:")
print("="*60)

import random
sample_dir = 'cropped_beetles/images_no_flash'
for genus_folder in sorted(os.listdir(sample_dir))[:3]:  # Primeras 3 carpetas
    genus_path = os.path.join(sample_dir, genus_folder)
    if os.path.isdir(genus_path):
        files = [f for f in os.listdir(genus_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            sample = random.sample(files, min(2, len(files)))
            print(f"\n📂 {genus_folder}:")
            for f in sample:
                print(f"   ✅ {f}")
