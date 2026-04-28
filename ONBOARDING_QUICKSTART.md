# TAMformer Onboarding Quickstart

Bu rehber, projeyi ilk kez açan birinin 10-15 dakikada sistemi anlaması ve ilk train/test çalıştırmasını yapması için hazırlandı.

## 1) Proje Ne Yapıyor?

- Task: yayanın anlık motion sınıfını tahmin eder.
- Girdi: `box + local_context`
- Zaman: son `1 saniye` (current frame dahil)
- Çıkış: 5 sınıf
  - `standing`, `walking`, `starting_to_move`, `running`, `stopping`

## 2) Kritik Dosyalar

- `run.py` -> ana giriş (train/test/resume)
- `tamformer.py` -> model mimarisi
- `data_generator.py` -> veri hazırlama + batch + label
- `jaad_data.py`, `pie_data.py` -> dataset parsing + class mapping
- `configs/configs_all.yaml`, `configs/configs_beh.yaml`, `configs/configs_pie.yaml` -> deney ayarları

## 3) Kurulum Kontrol Listesi

1. Python ortamını aktif et.
2. Dataset klasör yollarını config dosyasında doğrula:
   - `data_opts.path_to_dataset`
3. İlgili config seç:
   - JAAD all: `configs/configs_all.yaml`
   - JAAD beh: `configs/configs_beh.yaml`
   - PIE: `configs/configs_pie.yaml`

## 4) İlk Çalıştırma

### Train
```bash
python run.py --config_file configs/configs_all.yaml
```

### Test
```bash
python run.py --config_file configs/configs_all.yaml --test
```

### Resume
```bash
python run.py --config_file configs/configs_all.yaml --resume
```

## 5) Önemli Konfig Parametreleri

- `model_opts.obs_input_type`: `[box, local_context]`
- `model_opts.obs_seconds`: `1`
- `model_opts.num_classes`: `5`
- `model_opts.classifier_activation`: `softmax`
- `model_opts.classifier_loss`: `sparse_categorical_crossentropy`
- `model_opts.class_weights`: sınıf ağırlıkları

## 6) Beklenen Train/Eval Davranışı

- Model tek bir `softmax(5)` çıktı üretir.
- Eğitim kaybı multiclass sparse categorical akışındadır.
- Değerlendirme `argmax` tabanlıdır.
- Metrikler:
  - Accuracy
  - F1 (macro, weighted)
  - Precision (macro)
  - Recall (macro)
  - AUC macro OVR (uygun olduğunda)

## 7) Hızlı Debug İpuçları

- `shape mismatch` hatası alırsan:
  - `obs_input_type` ve `feat_size` eşleşmesini kontrol et.
- Dataset okunmuyorsa:
  - `path_to_dataset` yolunu doğrula.
- Sınıf dağılımı çok dengesizse:
  - `class_weights` değerlerini güncelle.

## 8) Yeni Katkı Yapacaklar İçin İlk Task Önerileri

1. `running` sınıfını daha güvenilir yapmak için hız-temelli heuristik ekleme.
2. Train log çıktısını CSV/TensorBoard ile daha görünür hale getirme.
3. Config validasyonu (eksik/uyumsuz parametrelerde erken hata) ekleme.

## 9) Nereden Devam Etmeli?

- Önce `PROJECT_OVERVIEW_FINAL.md` dosyasını oku (genel mimari).
- Sonra `IMPLEMENTATION_SUMMARY_TR_EN.md` dosyasını oku (yapılan dönüşüm detayları).
- Son olarak kendi dataset/config ile kısa bir train + test çalıştır.
