# gene sequence de-redundancy  
English verson:  
Gene sequence de-redundancy is a precise gene sequence de-redundancy software that supports heterogeneous acceleration.  
The existing de-redundancy software cd-hit is a CPU based software,
In order to make the running speed acceptable, increase the speed by reducing the accuracy.
Compared with accurate clustering, the speed can be up to 100 times faster, but the generated results contain redundancy.  
Gene sequence de-redundancy supports heterogeneous acceleration,
The generated result does not contain any redundancy, and the speed is the same as cd-hit.  
Compile:  
make  
Run:  
./cluster [options]  
Parameter Description:  
t: similarity threshold, the default is 0.95, range is 0.8 to 1.0.  
i: input file, default is testData.fasta  
o: output result file, default is result.fasta  
DevCloud platform running code:  
qsub submit.sh  

中文版：  
gene sequence de-redundancy是支持异构加速的精确基因序列去冗余软件。  
现有的去冗余软件cd-hit是CPU软件，
为了使得运行速度可接受，通过降低精度，来提高速度。
与精确聚类相比，速度最高可提升百倍，但生成的结果中包含冗余。  
gene sequence de-redundancy支持异构加速，
生成的结果中不包含任何冗余，而速度与cd-hit相同。  
编译代码：  
make  
运行代码:  
./cluster [选项]  
参数说明：  
t: 相似度阈值，默认为 0.95，支持0.8到1.0  
i: 输入文件，默认为 testData.fasta  
o: 输出结果文件，默认为 result.fasta  
devCloud平台运行代码：  
qsub submit.sh  
