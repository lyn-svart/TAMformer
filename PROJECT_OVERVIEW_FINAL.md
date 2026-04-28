# TAMformer Project Overview (Final State)

Bu doküman, projenin mevcut (güncel) halini ve genel yapısını hızlıca anlamak için hazırlanmıştır.

## 1) Projenin Güncel Amacı

Repo, TAMformer tabanlı yaya hareket tahmini modeli içerir. Son güncellemelerle birlikte sistem şu hedefe göre çalışır:

- Girdi modaliteleri: `box` + `local_context`
- Zaman penceresi: mevcut frame dahil son `1 saniye` (FPS'e göre dinamik)
- Çıkış: tek adımda `5 sınıf` motion tahmini
  - `standing`
  - `walking`
  - `starting_to_move`
  - `running`
  - `stopping`

## 2) Üst Düzey Akış

1. `run.py` config dosyasını okur.
2. Dataset arayüzleri (`jaad_data.py`, `pie_data.py`) ham track ve etiketleri üretir.
3. `data_generator.py`:
   - 1 saniyelik pencereyi çıkarır,
   - gerekli modaliteleri hazırlar,
   - tek sınıf etiketi üretir.
4. `tamformer.py` modeli kurar:
   - multimodal transformer encoder
   - tek `softmax(5)` prediction head
5. `run.py` modeli eğitir ve değerlendirir.

## 3) Dosya Bazlı Mimari

### `run.py`
- Ana giriş noktasıdır.
- Config yükleme, dataset hazırlama, model oluşturma, train/test döngüsünü yönetir.
- Multiclass loss/metric akışı bu dosyada tanımlanır.

### `tamformer.py`
- Model mimarisi burada tanımlıdır.
- Öğrenilebilir attention mask + transformer blokları içerir.
- Son katman tek head: `Dense(num_classes, activation='softmax')`.

### `data_generator.py`
- DataGetter: ham veriyi modele uygun tensörlere dönüştürür.
- DataGenerator: batch üretir.
- Dinamik gözlem penceresi (`obs_seconds` + FPS) ve tek hedef sınıf etiketi burada işlenir.

### `jaad_data.py` / `pie_data.py`
- Dataset parsing ve sequence üretimi.
- Ham davranış etiketlerini canonical 5 sınıfa mapleme mantığı içerir.

### `configs/*.yaml`
- Deney ayarlarını içerir.
- Güncel kurulumda:
  - `obs_input_type: [box, local_context]`
  - `num_classes: 5`
  - `obs_seconds: 1`
  - `classifier_activation: softmax`
  - `classifier_loss: sparse_categorical_crossentropy`

### `advanced_activations.py`
- Projenin özel aktivasyon/masking davranışı için eklenen destek dosyasıdır.

## 4) Veri ve Etiket Mantığı (Güncel)

- Sekanslardan son 1 saniye alınır (current frame dahil).
- Etiket, tek adım için sınıf-id olarak üretilir (sparse).
- Binary crossing yaklaşımından multiclass motion yaklaşımına geçilmiştir.

## 5) Eğitim ve Değerlendirme

- Loss: weighted sparse categorical crossentropy
- Tahmin: `argmax` ile sınıf seçimi
- Metrikler:
  - Accuracy
  - F1 (macro, weighted)
  - Precision (macro)
  - Recall (macro)
  - AUC (macro OVR, uygun olduğunda)

## 6) Çalıştırma (Pratik)

Örnek komutlar:

- Train:
  - `python run.py --config_file configs/configs_all.yaml`
  - `python run.py --config_file configs/configs_beh.yaml`
  - `python run.py --config_file configs/configs_pie.yaml`

- Test:
  - `python run.py --config_file configs/configs_all.yaml --test`

- Resume:
  - `python run.py --config_file configs/configs_all.yaml --resume`

## 7) Notlar ve Sınırlar

- `running` sınıfı bazı veri kaynaklarında doğrudan etiketlenmediği için mapleme fallback/transition mantığı içerir.
- Daha güçlü bir `running` tanımı gerekirse bbox tabanlı hız/yer değiştirme heuristiği eklenebilir.
- Bu repo, orijinal paper kod tabanı üzerine task-odaklı refactor içerir; README'deki bazı eski komut/adlar tarihsel kalmış olabilir.

---

## English Snapshot

- Final task: single-step 5-class pedestrian motion classification.
- Inputs: `box + local_context` only (pose removed).
- Temporal context: last 1 second including current frame (FPS-driven).
- Core modules:
  - `run.py`: orchestration (train/test/eval)
  - `tamformer.py`: model definition
  - `data_generator.py`: temporal slicing + batching + labels
  - `jaad_data.py`, `pie_data.py`: dataset parsing and class mapping
  - `configs/*.yaml`: experiment and model settings
