# BirdCLEF 2026 Data Setup

This project uses **Google Colab** for initial dataset acquisition and **Google Drive / My Drive** for persistent storage.

## Storage Layout

The setup notebook creates and uses this Drive layout:

```text
MyDrive/
`- birdclef-2026/
   |- raw/
   |  `- kaggle-download/
   |- data/
   |  |- train_audio/
   |  |- train_soundscapes/
   |  |- test_soundscapes_placeholder/
   |  |- train.csv
   |  |- taxonomy.csv
   |  |- sample_submission.csv
   |  |- train_soundscapes_labels.csv
   |  `- recording_location.txt
   |- outputs/
   |  |- checkpoints/
   |  |- oof/
   |  `- submissions/
   `- docs/
```

## Notebook

Run the Colab notebook at [../notebooks/00_setup_data_to_drive.ipynb](../notebooks/00_setup_data_to_drive.ipynb).

The notebook:

- mounts Google Drive
- installs Kaggle inside Colab if needed
- uploads or reads `kaggle.json`
- downloads `birdclef-2026` into Colab runtime storage
- preserves the raw Kaggle archive in Drive
- extracts the archive into runtime storage
- syncs the extracted data into Drive
- verifies the expected file manifest

## Notes

- Hidden `test_soundscapes/` audio is not expected to be available from the initial competition download.
- Use Drive as the persistent source of truth, but copy data into local runtime storage when training in Colab if Drive I/O is slow.
- The repository ignores local `data/` and `outputs/` contents by default so large artifacts do not get committed.
