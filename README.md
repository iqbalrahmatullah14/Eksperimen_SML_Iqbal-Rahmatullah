# Eksperimen SML - Iqbal Rahmatullah

Submission Kriteria 1 "Melakukan Eksperimen terhadap Dataset Pelatihan" — kelas Membangun Sistem Machine Learning (Dicoding).

Dataset: [Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) — 7043 baris, target biner `Churn`.

## Struktur

```
Eksperimen_SML_Iqbal-Rahmatullah
├── .github/workflows/preprocessing.yml   # CI otomatisasi preprocessing (Advance)
├── namadataset_raw/                      # dataset mentah
├── preprocessing/
│   ├── Eksperimen_Iqbal-Rahmatullah.ipynb   # notebook eksperimen (data loading, EDA, preprocessing manual)
│   ├── automate_Iqbal-Rahmatullah.py        # preprocessing otomatis (Skilled)
│   └── telco_customer_churn_preprocessing/  # hasil train.csv & test.csv
└── requirements.txt
```

## Menjalankan otomatisasi manual

```bash
pip install -r requirements.txt
python preprocessing/automate_Iqbal-Rahmatullah.py
```

GitHub Actions (`.github/workflows/preprocessing.yml`) menjalankan skrip yang sama setiap kali `namadataset_raw/` berubah atau lewat trigger manual (`workflow_dispatch`), lalu mengembalikan dataset terbaru yang sudah diproses.
