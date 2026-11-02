# PneumoniaMNIST – CSL Project  
Confronto tra **SVM (PCA/HOG)** e **CNN (SGD vs Adam)**  
per la classificazione di immagini mediche (MedMNIST v2)

## Descrizione del progetto
Questo progetto implementa, in un unico file compatto (`pneumonia.py`),  
un confronto tra metodi classici di machine learning (SVM)  
e modelli di deep learning (CNN) sul dataset **PneumoniaMNIST**.

Il codice è strutturato a blocchi, nello stesso stile accademico  
utilizzato nei progetti della docente:

- Caricamento dataset  
- Conversione in NumPy (per SVM)  
- Pipeline SVM + PCA  
- Pipeline SVM + HOG  
- Definizione CNN compatta  
- Training CNN con SGS + Momentum  
- Training CNN con Adam  
- Valutazione e salvataggio risultati  

Tutti i risultati vengono salvati nella cartella `results_compact/`.

---

## Requisiti

Installare le dipendenze da `requirements.txt`:

```bash
pip install -r requirements.txt
