# gene sequence de-redundancy  
This is rocm verson.  
Compile:  
hipcc main.cpp func.cpp -o cluster  
Run:  
./cluster [options]  
Parameter Description:  
t: similarity threshold, the default is 0.95, range is 0.8 to 1.0.  
i: input file, default is testData.fasta  
o: output result file, default is result.fasta  
