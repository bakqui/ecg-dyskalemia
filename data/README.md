# ECG Dataset Preparation

To run the code, the dataset needs to be prepared and satisfy a couple of conditions.
- ECG signals needs to be `numpy.ndarray` or `Dict[str, numpy.ndarray]` and be __saved as PKL files__.
  - For a dict type ECG waveform, it must have following keys, standard 12-lead ECG lead names.
    - `I`
    - `II`
    - `III`
    - `AVR`
    - `AVL`
    - `AVF`
    - `V1`
    - `V2`
    - `V3`
    - `V4`
    - `V5`
    - `V6`
- There must be a `pandas.DataFrame` __index file which contains the information of the dataset__.
  1) A list of ECG file names.
  2) A list of ECG sampling rates if the signals have different sampling rate.
  3) A list of labels of ECGs.

We present dummy ECG signals (`./demo/ecgs/###.pkl`) and index file (`./demo/index.csv`) as a demo.
