# cd-hit-est-by-OneAPI  
移植了cd-hit套件中的cd-hit-est应用到OnePAI平台  

编译代码：  
dpcpp cd-hit-set.cpp
devCloud平台运行代码：  
./q run.sh  

由于程序处于初期阶段，还有些粗糙的地方，说明如下：  
移植自cpu程序cd-hit套件的cd-hit-est工具，功能是基因序列聚类。  
默认聚类阈值为0.9，word长度为8，可以在源码define处修改。  
性能比原cpu版差很多，后期会优化。  
测试文件testData.fasta需要与编译生成的可执行文件a.out在同一目录。  
测试文件大概需要跑五分钟，只会输出聚类的条数，实际聚类结果存在main函数的clusters数组中，没有输出。  
测试文件的文件名是写在源码中的。  
