# Intelligent source separation system for audio-domain signals
This git repository contains the source codes for the Diploma Thesis. The scripts created are as follows:
- `apply_wiener_filter.py`: applies the Multichannel Wiener Filter to a whole directory, structured the same as the MUSDB18 dataset one. Implements the Data post-processing step of the Implementation pipeline;
- `check_artifacts.py`: for a given track's source from the dataset, vizualizes the frequency spectrum of the ground truth, first estimation and filtered source in order to find the frequency artifacts introduced by the filter;
- `custom_callbacks.py`: implements various custom callbacks for the training process;
- `custom_initializers.py`: implements a custom weight initializer for the models;
- `custom_layers.py`: implements 2 custom layers for the Open Unmix model, which was the first attempt of a Source Separation model for this project;
- `custom_losses.py`: implements various custom losses for models;
- `get_results.py`: given a directory of source estimations, generates the metrics and stores them in a *.csv* file;
- `imports.py`: imports every library needed for the main pipeline;
- `lr_schedulers.py`: implements a custom learning rate scheduler for the models;
- `main.py`: the main script of this project, it executes the Data pre-processing, Training and Inference steps of the Implementation pipeline;
- `models.py`: implements different Neural Network models for Source Separation, as well as a WaveNet architecture for audio signal enhancement, which was not succesfull, unfortunately;
- `normalize_dataset.py`: normalizez the whole directory of source estimations;
- `open_unmi_extract_mean_and_std.py`: for the Open Unmix model, extract the mean and standard deviation of the traing set of the dataset for standardization;
- `parse_dataset.py`: iterates over the whole dataset, saving different statistical information, such as the minimum, maximum, mean, std, etc.;
- `predict_database.py`: generates first source estimates for every track of the dataset;
- `predict_many_models.py`: generates different source estimates from multiple models, for a given track;
- `prediction.py`: implements 2 versions of the Inference step of the Implementation pipeline;
- `prediction_source_phase.py`: implements a function that make a source estimate, but uses the phase of the original source, for testing purposes;
- `prepoc.py`: smaller, more compact version of `preprocess.py`, for testing purposes;
- `preprocess.py`: implements different versions of the Data pre-processing step of the Implementation pipeline;
- `resampling.py`: resamples an audio file and resaves it;
- `resume_training.py`: function that resumes the training process of a model;
- `SNR.py`: computes the Signal-to-Noise Ratio for a whole directory of estimations;
- `training.py`: implements the Trainig step of the Implementation pipeline;
- `utils.py`: implements different useful functions;
- `utils2.py`: smaller version of `utils.py`, for testing purposes;

