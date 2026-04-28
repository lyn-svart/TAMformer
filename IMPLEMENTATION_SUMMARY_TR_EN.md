# TAMformer 5-Class Single-Step Conversion Summary

## TR - Yapılan Değişiklikler

Bu doküman, TAMformer projesinde yapılan "5 sınıf, tek-adım motion tahmini" dönüşümünü özetler.

### 1) Hedef
- Model artık `pose` kullanmıyor.
- Girdi modaliteleri: `box + local_context`.
- Zaman penceresi: mevcut frame dahil geçmiş `1 saniye` (FPS tabanlı dinamik hesap).
- Çıkış: tek head ile 5 sınıf motion tahmini:
  - `0: standing`
  - `1: walking`
  - `2: starting_to_move`
  - `3: running`
  - `4: stopping`

### 2) Güncellenen Dosyalar
- `configs/configs_all.yaml`
- `configs/configs_beh.yaml`
- `configs/configs_pie.yaml`
- `jaad_data.py`
- `pie_data.py`
- `data_generator.py`
- `tamformer.py`
- `run.py`

### 3) Konfigürasyon Değişiklikleri
- `obs_input_type` içinden `pose` çıkarıldı.
- `feat_size` değerleri `box + local_context` için yeniden hizalandı.
- `num_classes: 5` eklendi.
- `obs_seconds: 1` eklendi (runtime'da `obs_length = fps * obs_seconds`).
- `classifier_activation` -> `softmax`
- `classifier_loss` -> `sparse_categorical_crossentropy`
- `class_weights` alanı 5 sınıfı destekleyecek şekilde eklendi.

### 4) Veri Etiketi ve Sınıf Haritalama
- JAAD ve PIE için canonical motion map eklendi.
- Binary crossing etiketi yerine 5-sınıf motion etiketi üretimi yapılıyor.
- Etiket üretimi `activities` akışına entegre edildi.

### 5) 1 Saniye Pencere + Tek Hedef
- `data_generator.py` içinde örnekleme mantığı güncellendi:
  - Sekanslar son `obs_length` frame'e kırpılıyor.
  - Sekans kısa ise baştan pad edilerek `obs_length` tamamlanıyor.
- Çok-adımlı label çoğaltma kaldırıldı.
- Generator artık tek class-id (`sparse`) hedef döndürüyor.
- Eski binary pos/neg batch ayrımı kaldırıldı (multiclass uyum).

### 6) Model Mimarisi
- Hardcoded `136` ve `40` bağımlılıkları kaldırıldı.
- Girdi uzunluğu `obs_length` üzerinden dinamik hale getirildi.
- 40 adet sigmoid çıkış yerine tek `Dense(num_classes, softmax)` head kullanılıyor.

### 7) Eğitim ve Değerlendirme
- Binary ağırlıklı kayıp yerine multiclass sparse categorical kayıp kullanılıyor.
- Class weighting 5 sınıf için çalışacak şekilde güncellendi.
- Test/eval akışı `argmax` tabanlı multiclass değerlendirmeye geçirildi.
- Raporlanan metrikler:
  - Accuracy
  - F1 (macro, weighted)
  - Precision (macro)
  - Recall (macro)
  - AUC (macro OVR, uygun olduğunda)

### 8) Doğrulama
- Güncellenen Python dosyaları derleme/sözdizimi kontrolünden geçti (`py_compile`).
- Düzenlenen dosyalarda linter hatası bulunmadı.

### 9) Not
- `running` sınıfı bazı veri seti etiketlerinde doğrudan bulunmadığı için fallback/transition mantığı ile haritalanmıştır. Gerekirse bu kısım hız-temelli (bbox displacement) daha güçlü bir kurala çekilebilir.

---

## EN - What Was Implemented

This document summarizes the "5-class, single-step motion prediction" refactor applied to TAMformer.

### 1) Goal
- The model no longer uses `pose`.
- Input modalities are now: `box + local_context`.
- Temporal window is `1 second` of history including the current frame (FPS-based dynamic length).
- Output is a single prediction head with 5 motion classes:
  - `0: standing`
  - `1: walking`
  - `2: starting_to_move`
  - `3: running`
  - `4: stopping`

### 2) Updated Files
- `configs/configs_all.yaml`
- `configs/configs_beh.yaml`
- `configs/configs_pie.yaml`
- `jaad_data.py`
- `pie_data.py`
- `data_generator.py`
- `tamformer.py`
- `run.py`

### 3) Configuration Changes
- Removed `pose` from `obs_input_type`.
- Realigned `feat_size` for `box + local_context`.
- Added `num_classes: 5`.
- Added `obs_seconds: 1` (runtime computes `obs_length = fps * obs_seconds`).
- Switched `classifier_activation` to `softmax`.
- Switched `classifier_loss` to `sparse_categorical_crossentropy`.
- Added `class_weights` for 5-class support.

### 4) Labeling and Class Mapping
- Added a canonical 5-class motion mapping for JAAD and PIE.
- Replaced binary crossing label generation with 5-class motion labels.
- Integrated this mapping into the `activities` label pipeline.

### 5) 1-Second Window + Single Target
- Updated sequence preparation in `data_generator.py`:
  - Sequences are trimmed to the last `obs_length` frames.
  - Short sequences are front-padded to match `obs_length`.
- Removed multi-step label replication.
- Generator now returns a single sparse class-id target.
- Removed old binary pos/neg balanced batching path (multiclass-compatible flow).

### 6) Model Architecture
- Removed hardcoded temporal assumptions (`136` and `40`).
- Input timeline is now dynamic via `obs_length`.
- Replaced 40 sigmoid outputs with a single `Dense(num_classes, softmax)` head.

### 7) Training and Evaluation
- Replaced weighted binary loss with weighted multiclass sparse categorical loss.
- Updated class weighting logic for 5 classes.
- Migrated evaluation to `argmax`-based multiclass scoring.
- Reported metrics now include:
  - Accuracy
  - F1 (macro, weighted)
  - Precision (macro)
  - Recall (macro)
  - AUC (macro OVR, when applicable)

### 8) Validation
- Updated Python files pass syntax/compile checks (`py_compile`).
- No linter errors in the edited files.

### 9) Note
- Since `running` is not explicitly available in all dataset annotations, it is currently mapped via fallback/transition logic. This can be replaced by a stronger speed-based heuristic (e.g., bbox displacement threshold) if needed.
