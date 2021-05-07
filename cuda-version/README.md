# gene sequence de-redundancy  
This is cuda verson.  
Compile:  
nvcc main.cu func.cu -o cluster -DMAX_LENGTH=3000  
MAX_LENGTH: length of the longest sequence in data set  
Run:  
./cluster [options]  
Parameter Description:  
t: similarity threshold, the default is 0.95, range is 0.8 to 1.0  
i: input file, default is testData.fasta  
o: output result file, default is result.fasta  

# For example  
./cluster t 0.85 i testData.fasta o result.fasta  
