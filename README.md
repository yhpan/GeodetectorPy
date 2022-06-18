# GeodetectorPy
Geodetector is a software for measure and attribution of/by spatial stratified heterogeneity (SSH), which is developed by Prof. Wang Jingfeng. There are R and Excel version. Details in http://geodetector.cn

Here are the python script. The script are under development and not a offical version. please be caution when you use it.
the main script is in path 'GeodetectorPy'
in path 'data', there are three .csv file contain sample data from http://geodetector.cn, and geotiff data in './data/smpledata2019EVILC'  
please refer the demo.py to learn how use the script


Spatial stratified heterogeneity (SSH), referring to the within strata are more similar than the between strata, a model with global parameters would be confounded if input data is SSH. 
Note that the "spatial" here can be either geospatial or the space in mathematical meaning. 
Geographical detector is a novel tool to investigate SSH: 
(1) measure and find SSH of a variable Y; 
(2) test the power of determinant X of a dependent variable Y according to the consistency between their spatial distributions;
(3) investigate the interaction between two explanatory variables X1 and X2 to a dependent variable Y 
(Wang et al 2014 <doi:10.1080/13658810802443457>, Wang, Zhang, and Fu 2016 <doi:10.1016/j.ecolind.2016.02.052>).