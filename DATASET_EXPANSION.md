# Dataset Expansion Notes

The current dataset contains:

- `4,920` training rows
- `41` disease classes
- `147` symptom columns

The generated file `csv_files/Training_augmented_2m.csv.gz` contains `2,000,000` augmented rows created from the original labeled symptom patterns.

Important: this file increases the number of training examples. It does not add clinically validated new disease classes. To add new diseases or thousands of symptoms safely, merge a verified medical dataset that includes labeled disease/symptom relationships, then rerun preprocessing and training.

## Generated Files

- `s00_expand_dataset.py`: creates the large augmented dataset.
- `csv_files/Training_augmented_2m.csv.gz`: compressed 2M-row training file.
- `csv_files/Training_augmented_2m_metadata.json`: metadata and safety note.

## Regenerate

```powershell
python .\s00_expand_dataset.py --rows 2000000 --chunk-size 100000
```

## Use In Preprocessing

PowerShell:

```powershell
$env:DISEASE_TRAINING_CSV="csv_files/Training_augmented_2m.csv.gz"
python .\s01_prep.py
```

Then train a model:

```powershell
python .\s03_train.py
```

Note: the current saved model is K-Nearest Neighbors. KNN can become slow with millions of rows, so for real use on the augmented dataset, prefer a tree model or train on a sampled/validated dataset.
