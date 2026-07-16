# Predikcija kategorije proizvoda na osnovu naslova

Ovaj projekat koristi mašinsko učenje (Logističku regresiju) za automatsku klasifikaciju proizvoda u odgovarajuće kategorije na osnovu njihovog tekstualnog naslova.

## Struktura projekta
* "products.csv" - Skup podataka korišćen za treniranje.
* "modelovanje.ipynb" - Jupyter sveska sa kompletnom analizom i razvojem rešenja.
* "train_model.py" - Python skripta za treniranje modela i čuvanje na disk.
* "predict_category.py" - Python skripta za interaktivno testiranje modela kroz terminal.
* "README.md" - Uputstvo za pokretanje projekta.

## Uputstvo za pokretanje i testiranje
1. **Treniranje modela:**
    Pokrenite skriptu za treniranje kako bi se generisale binarne datoteke modela:
    ```bash
    python train_model.py