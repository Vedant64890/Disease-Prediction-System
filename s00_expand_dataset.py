"""
Create a large augmented disease/symptom training dataset.

This does not invent new medical diseases. It expands the existing labeled
Training.csv patterns into many varied rows for experimentation.
Use clinically validated external data before adding new disease classes.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("csv_files/Training.csv")
DEFAULT_OUTPUT = Path("csv_files/Training_augmented_2m.csv.gz")
DEFAULT_METADATA = Path("csv_files/Training_augmented_2m_metadata.json")


def load_training_data(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()
    if "prognosis" not in df.columns:
        raise ValueError("Training data must include a 'prognosis' column.")

    symptom_cols = [col for col in df.columns if col != "prognosis"]
    df[symptom_cols] = df[symptom_cols].fillna(0).astype(np.uint8)
    df["prognosis"] = df["prognosis"].astype(str).str.strip()
    return df, symptom_cols


def build_class_stats(df, symptom_cols):
    diseases = sorted(df["prognosis"].unique())
    disease_to_index = {disease: idx for idx, disease in enumerate(diseases)}
    row_disease_idx = df["prognosis"].map(disease_to_index).to_numpy()
    base_x = df[symptom_cols].to_numpy(dtype=np.uint8)
    base_y = df["prognosis"].to_numpy()

    prevalence = np.zeros((len(diseases), len(symptom_cols)), dtype=np.float32)
    for disease, disease_idx in disease_to_index.items():
        disease_rows = base_x[row_disease_idx == disease_idx]
        prevalence[disease_idx] = disease_rows.mean(axis=0)

    return diseases, disease_to_index, row_disease_idx, base_x, base_y, prevalence


def generate_augmented_chunk(base_x, base_y, row_disease_idx, prevalence, chunk_size, rng):
    chosen_rows = rng.integers(0, len(base_y), size=chunk_size)
    x_chunk = base_x[chosen_rows].copy()
    y_chunk = base_y[chosen_rows]
    disease_idx = row_disease_idx[chosen_rows]
    disease_prevalence = prevalence[disease_idx]

    active_mask = x_chunk == 1
    core_symptom_mask = disease_prevalence >= 0.85
    drop_mask = active_mask & (~core_symptom_mask) & (rng.random(x_chunk.shape) < 0.10)
    x_chunk[drop_mask] = 0

    add_probability = np.clip((disease_prevalence * 0.08) + 0.0015, 0.0, 0.10)
    add_mask = (x_chunk == 0) & (disease_prevalence > 0) & (rng.random(x_chunk.shape) < add_probability)
    x_chunk[add_mask] = 1

    random_noise_mask = (x_chunk == 0) & (rng.random(x_chunk.shape) < 0.0008)
    x_chunk[random_noise_mask] = 1

    too_sparse = np.where(x_chunk.sum(axis=1) == 0)[0]
    if len(too_sparse):
        x_chunk[too_sparse] = base_x[chosen_rows[too_sparse]]

    return x_chunk, y_chunk


def write_augmented_dataset(input_path, output_path, metadata_path, target_rows, chunk_size, seed):
    df, symptom_cols = load_training_data(input_path)
    diseases, _, row_disease_idx, base_x, base_y, prevalence = build_class_stats(df, symptom_cols)
    rng = np.random.default_rng(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    first_chunk = True
    with gzip.open(output_path, "wt", encoding="utf-8", newline="", compresslevel=1) as f:
        while rows_written < target_rows:
            current_chunk_size = min(chunk_size, target_rows - rows_written)
            x_chunk, y_chunk = generate_augmented_chunk(
                base_x,
                base_y,
                row_disease_idx,
                prevalence,
                current_chunk_size,
                rng,
            )
            chunk_df = pd.DataFrame(x_chunk, columns=symptom_cols)
            chunk_df["prognosis"] = y_chunk
            chunk_df.to_csv(f, index=False, header=first_chunk)
            rows_written += current_chunk_size
            first_chunk = False
            print(f"Wrote {rows_written:,}/{target_rows:,} rows")

    metadata = {
        "source_file": str(input_path),
        "output_file": str(output_path),
        "target_rows": target_rows,
        "original_rows": int(len(df)),
        "disease_classes": len(diseases),
        "symptom_columns": len(symptom_cols),
        "diseases": diseases,
        "augmentation_note": (
            "Synthetic rows generated from existing labeled symptom patterns. "
            "This file increases training examples, not clinically validated disease coverage."
        ),
        "seed": seed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a large augmented disease/symptom CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = write_augmented_dataset(
        args.input,
        args.output,
        args.metadata,
        args.rows,
        args.chunk_size,
        args.seed,
    )
    print("\nDataset expansion complete.")
    print(f"Output: {summary['output_file']}")
    print(f"Rows: {summary['target_rows']:,}")
    print(f"Diseases: {summary['disease_classes']}")
    print(f"Symptoms: {summary['symptom_columns']}")
