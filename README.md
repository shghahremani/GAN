In this repository we have developed two models as following:  
1- We used generative adversarial networks (GANs) to do anomaly detection for time series data.   
2- We also developed attention based LSTM to identify time delay attack attributes.  

# Data

We have two kind of dataset as following:  

# DataSet for anomaly detection (Gan Network):

Training the model:  
In order to train GAN network, first we need to train the model with normal data which has been generated by Powerword simulator. In this case the data only contains the natural noise. Please go to the GAN/dataset/GAN_training.csv directory to access to this dataset. In this dataset data does not any label.  

Testing GAN model:  

Dataset for testing the trained GAN network contains both the normal and anomaly ones. Anomaly data means that data is collected when the system attacked by time delay attack at different point and various length. Here, data is labelled with normal(0) and anomaly(1).

# DataSet for time delay attack identification (attention based GRU/LSTM)
 You can download the Dataset for training <a href="https://drive.google.com/open?id=1R9oDjmwUL5A_mfs7Z6-PBlKrlhqB7zIN">here</a> and for the testing  <a href="https://drive.google.com/open?id=1ssG18uFcQRMb4-3t2xDoVRpL2_SBQ9ob">here</a>
Training and testing the model:  
Training and testing the model are done with data which is labelled with time delay magnitude and also attack lunch point. You may find these   

# Data description

# DataSet for anomaly detection (Gan Network):

#Training dataset:   
Each record of the dataset implies the system sensors value with various settings for 400 seconds which contains the following columns:  
Randomload : This column contains the natural noise for each experiment. This column has nothing to do with the training and it should be ignored while training the model. Each record(each experiment contain) is simulated with this natural noise.    
ACE_1 : area 1 control error (ACE)   
ACE_2 : area 2 control error (ACE)  
ACE_3 : area 3 control error (ACE)  
DemandArea_1: power demand of area 1  
DemandArea_2: power demand of area 2  
DemandArea_3: power demand of area 3  
ExportArea_1: power Export of area 1  
ExportArea_2: power Export of area 2  
ExportArea_3: power Export of area 3  
sys_Freq : system Frequency  
ROCOF_3: Rate of change of frequency for line #3  
We have 37 line in the 37-bus IEEE system. So we have 37 ROCOF.   
Tiline_1: A transmission line connecting two buses belonging to two areas is called a tie-line.  
We have 8 tieline in the 37 BUS IEEE system.  
# NOTE: 
As we run the Powerworld simulator for 400 second with time step of 4, so each sensor value has 100 values from 1 to 100. For example for ACE_1 we have 100 consecutive value for each record of the dataset. After that 100 values for ACE_2 are appeared in the dataset.  
# Testing dataset for GAN network:
The columns for the testing the GAN model is similar to the training GAN dataset, however, here data is not necessarily normal and it may be anomaly. So, here, each record of the dataset is labelled with 0 (for those which are normal and 1 for anomalous ones).  
# Dataset description of attention based GRU/LSTM:
delayStart: attack start point  
delayEnd: end point of attack  
delayLengh: length of attack  
delayCycle: number of cycle which each packet is delayed during the attack time  
unsafety_state: this columns show whether system is safe(0) or unsafe(1).  
unsafety_time: the moment when system goes to unsafe zone  
sysDown_state: this columns show whether system is stable(0) or unstable(1).  
sysDown_time: the moment when system goes to unstable zone  
The rest of columns are similar to the GAN dataset.  







