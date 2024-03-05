## Models Developed:
1. Utilized generative adversarial networks (GANs) for anomaly detection in time series data.
2. Developed an attention-based LSTM model to identify attributes of time delay attacks.

## Data:

### Dataset for Anomaly Detection (GAN Network):
- **Training Dataset:** The training dataset for the GAN network contains system sensor values with various settings over 400 seconds. It includes columns such as ACE_1, ACE_2, DemandArea_1, DemandArea_2, DemandArea_3, ExportArea_1, ExportArea_2, ExportArea_3, sys_Freq, ROCOF_3, and Tiline_1. The 'Randomload' column represents natural noise and should be ignored during model training.

- **Testing Dataset:** The testing dataset includes normal and anomaly data labeled as 0 and 1 respectively. Each record in the testing dataset corresponds to system sensor values similar to the training dataset.

### Dataset for Time Delay Attack Identification (Attention-Based GRU/LSTM):
- **Training and Testing:** Data for training and testing the model is labeled with time delay magnitude and attack launch point. The dataset includes columns like delayStart, delayEnd, delayLength, delayCycle, unsafety_state, unsafety_time, sysDown_state, and sysDown_time.

### Data Description:
- **Note:** The Powerworld simulator runs for 400 seconds with a time step of 4, resulting in 100 consecutive values for each sensor parameter in the dataset.






