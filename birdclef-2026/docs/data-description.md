# BirdCLEF+ 2026 Dataset Description

## Header

Cornell Lab of Ornithology · Research Code Competition · 3 months to go  
BirdCLEF+ 2026  
Acoustic Species Identification in the Pantanal, South America

## Dataset Description

Your challenge in this competition is to identify which species (birds, amphibians, mammals, reptiles, insects) are calling in recordings made in the Brazilian Pantanal. This is an important task for scientists who monitor animal populations for conservation purposes. More accurate solutions could enable more comprehensive monitoring.

This competition uses a hidden test set. When your submitted notebook is scored, the actual test data will be made available to your notebook.

## Files

### `train_audio/`

The training data consists of short recordings of individual bird, amphibian, reptile, mammal, and insect sounds generously uploaded by users of xeno-canto.org and iNaturalist. These files have been resampled to 32 kHz where applicable to match the test set audio and converted to the `ogg` format.

Filenames consist of `[collection][file_id_in_collection].ogg`.

The training data should have nearly all relevant files; the organizers expect there is no benefit to looking for more on xeno-canto.org or iNaturalist and ask participants to limit burden on those servers. If additional collection is attempted, participants should follow the scraping rules of those data portals.

### `test_soundscapes/`

When you submit a notebook, the `test_soundscapes` directory will be populated with approximately 600 recordings used for scoring. These files are:

- 1 minute long
- in `ogg` format
- resampled to 32 kHz

The filenames have the general form:

`BC2026_Test_<file ID>_<site>_<date>_<time in UTC>.ogg`

Example:

`BC2026_Test_0001_S05_20250227_010002.ogg`

This example indicates:

- file ID `0001`
- site `S05`
- recorded on `2025-02-27`
- recorded at `01:00 UTC`

It should take a submission notebook approximately five minutes to load all test soundscapes.

Not all species from the training data actually occur in the test data.

### `train_soundscapes/`

Additional audio data from roughly the same recording locations as the `test_soundscapes`.

Filenames follow the same naming convention as the `test_soundscapes`. Although some recording sites overlap between train and test, precise recording dates and times do **not** overlap with recordings in the hidden test data.

This year, some `train_soundscapes` have been labeled by expert annotators. Ground truth is provided for a subset of these files in `train_soundscapes_labels.csv` with:

- `filename`: the soundscape file
- `start`: the start of the labeled 5-second segment
- `end`: the end of the labeled 5-second segment
- `primary_label`: a semicolon-separated list of species codes marked as present in that segment

Important note: some species occurring in the hidden test data might only have training samples in the labeled portion of `train_soundscapes` and not in `train_audio` (`XC` and `iNat` data). However, not all species from `train_soundscapes` occur in the `test_soundscapes`.

### `train.csv`

A wide range of metadata is provided for the training data. The most directly relevant fields are:

- `primary_label`: a code for the species (`eBird` code for birds, iNaturalist taxon ID for non-birds). Species information may be reviewed by appending codes to eBird or iNaturalist URLs, for example:
  - `https://ebird.org/species/brnowl` for Barn Owl
  - `https://www.inaturalist.org/taxa/41970` for Jaguar
- `secondary_labels`: list of species labels marked by recordists as also occurring in the recording; this can be incomplete
- `latitude` and `longitude`: coordinates where the recording was taken
- `author`: the user who provided the recording; `Unknown` if no name was provided
- `filename`: the associated audio filename
- `rating`: values in `1..5` from Xeno-canto (`1` low quality, `5` high quality, with `0.5` reduction when background species are present); `0` implies no rating is available; iNaturalist does not provide quality ratings
- `collection`: either `XC` or `iNat`, indicating the source collection

### `sample_submission.csv`

A valid sample submission.

- `row_id`: a slug of `[soundscape_filename]_[end_time]` for the prediction
- Example: segment `00:15-00:20` of test soundscape `BC2026_Test_0001_S05_20250227_010002.ogg` has row ID `BC2026_Test_0001_S05_20250227_010002_20`
- `[species_id]`: there are 234 species ID columns; you must predict the probability of presence for each species for every row

### `taxonomy.csv`

Data on the different species, including iNaturalist taxon ID and class name (`Aves`, `Amphibia`, `Mammalia`, `Insecta`, `Reptilia`).

Most insect species in this competition have not been identified at the species level and instead occur as sonotypes, for example `47158son16` for insect sonotype 16. These sonotypes are treated as classes despite lacking species IDs, and some also occur in the test data.

The 234 rows of this file represent the 234 class columns in the submission file. `primary_label` specifies the submission file column name.

### `recording_location.txt`

Some high-level information on the recording location: Pantanal, Brazil.

## Dataset Summary

- Files: 46,213
- Size: 16.14 GB
- Types: `ogg`, `csv`, `txt`
- License: `CC BY-NC-SA 4.0`

## Data Explorer Summary

- `test_soundscapes`
- `train_audio`
- `train_soundscapes`
- `recording_location.txt`
- `sample_submission.csv`
- `taxonomy.csv`
- `train.csv`
- `train_soundscapes_labels.csv`

Summary shown:

- 46.2k files
- 259 columns

## Download

### Kaggle CLI

```bash
kaggle competitions download -c birdclef-2026
```

### `kagglehub`

```python
kagglehub.competition_download('birdclef-2026')
```

## Metadata

- License: `CC BY-NC-SA 4.0`

## Access Note

To view or download the data, you need to agree to the competition rules and join the competition.
