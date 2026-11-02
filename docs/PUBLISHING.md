# Pubblicazione con GitHub Desktop

La cartella è preparata per la pubblicazione; non è stata creata né caricata una repository remota.

Se hai già creato la repository su GitHub, clonala da **File → Clone Repository → URL** in una nuova cartella locale. Copia il contenuto del progetto direttamente nella cartella clonata, includendo `.gitignore`, senza creare un'ulteriore sottocartella. In GitHub Desktop controlla **Changes**, crea il commit e premi **Push origin**.

Includi codice, README, requisiti, report `.tex` e `.pdf`, PDF delle slide originali e le cartelle `assets`, `results`, `scripts`, `archive`, `docs`. La cartella `results` contiene solo trascrizioni dei dati delle slide e metadati: non sono nuovi risultati di training.

Escludi `.venv` (l'ambiente locale è molto pesante), dati scaricati, checkpoint e cartelle di nuove esecuzioni. Il `.gitignore` lo fa automaticamente. Nel Finder mostra i file nascosti con **⌘ + ⇧ + .** per copiare anche `.gitignore`. Non copiare o modificare la cartella `.git` della repository clonata.

Il PowerPoint modificabile non viene pubblicato: il link alle slide nel README apre il PDF. Il report e i grafici si ricostruiscono senza il PowerPoint; lo script di estrazione opzionale richiede invece il file originale fornito a parte.

Il nuovo report si chiama `PneumoniaMNIST_MatteoLugli.pdf`. Il file già esistente `pneumonia_MatteoLugli.pdf` è l'esportazione delle slide ed è conservato come fonte.

Nome suggerito della repository: `pneumoniamnist-svm-cnn`.

Descrizione suggerita:

> SVM (PCA/HOG) and compact CNN (SGD/Adam) comparison on PneumoniaMNIST. Computational and Statistical Learning course project at UniMoRe.

Topic suggeriti: `machine-learning`, `pneumoniamnist`, `medmnist`, `svm`, `cnn`, `pytorch`, `scikit-learn`, `course-project`.

Messaggio per il primo commit:

```text
Add PneumoniaMNIST project, English report and presentation results
```

Se vuoi rendere esplicitamente riutilizzabile il materiale originale, scegli una licenza e aggiungi `LICENSE`. Mantieni separati i termini del dataset e delle dipendenze riportati in `THIRD_PARTY.md`.
