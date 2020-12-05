1. The application consists of two parts: the frond-end part is in visualization.twb file, which can be opened by Tableau. The back-end 
part is in project.jar file, which can be runned on Dumbo with command:
spark2-submit project.jar --deploy-mode cluster

2. The analytic result of the code is printed at the end of code execution, which is provided in screenshots folder. During the execution, four folders
are created. train and test folders contain the output data for training and testing the model; table and graph folders contain the output data for visualization.

3.the input data is stored in HDFS on Dumbo with path: jl7147/loudacre/cleanData