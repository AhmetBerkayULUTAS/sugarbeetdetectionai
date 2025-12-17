# Pancar tespit sonuçları
# Her tespit için bounding box koordinatları ve güven skoru
# int32 camera_id

int32 count                    # Toplam tespit sayısı
float32[] confidences          # Her tespitin güven skorları [0.85, 0.92, ...]
int32[] x1                     # Bounding box sol üst x koordinatları
int32[] y1                     # Bounding box sol üst y koordinatları  
int32[] x2                     # Bounding box sağ alt x koordinatları
int32[] y2                     # Bounding box sağ alt y koordinatları
int32[] class_ids              # Sınıf ID'leri (hepsi 0 = sugar_beet)
