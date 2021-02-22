#include "func.h"
//--------------------函数--------------------//
// printUsage 输出用法
void printUsage() {
    std::cout << "请校验输入参数"  << std::endl;
    std::cout << "a.out i inputFile t threshold" << std::endl;
    exit(0);  // 退出程序
}
// checkOption 检查输入
void checkOption(int argc, char **argv, Option &option) {
    if (argc%2 != 1) printUsage();  // 参数个数不对
    option.inputFile = "testData.fasta";  // 输入文件名
    option.outputFile = "result.fasta";  // 输出文件名
    option.threshold = 0.95;
    for (int i=1; i<argc; i+=2) {  // 遍历参数
        switch (argv[i][0]) {
        case 'i':
            option.inputFile = argv[i+1];
            break;
        case 'o':
            option.outputFile = argv[i+1];
            break;
        case 't':
            option.threshold = std::stof(argv[i+1]);
            break;
        default:
            printUsage();
            break;
        }
    }
    if (option.threshold < 0.8 || option.threshold >= 1) {
        std::cout << "阈值超出范围" << std::endl;
        exit(0);
    }
    int temp = (option.threshold*100-80)/5;
    switch (temp) {  // 根据阈值分配wordLength
    case 0:  // threshold:0.80-0.85 wordLength:5
        option.wordLength = 4;
        break;
    case 1:  // threshold:0.85-0.90 wordLength:6
        option.wordLength = 5;
        break;
    case 2:  // threshold:0.90-0.95 wordLength:7
        option.wordLength = 6;
        break;
    case 3:  // threshold:0.90-1.00 wordLength:8
        option.wordLength = 7;
        break;
    }
    std::cout << "输入文件:\t" << option.inputFile << std::endl;
    std::cout << "输出文件:\t" << option.outputFile << std::endl;
    std::cout << "相似阈值:\t" << option.threshold << std::endl;
    std::cout << "word长度:\t" << option.wordLength << std::endl;
}
// readFile 读文件
void readFile(std::vector<Read> &reads, Option &option) {
    std::ifstream file(option.inputFile);
    Read read;
    std::string line;
    getline(file, line);
    read.name = line;
    while (getline(file, line)) {  // getline不读换行符
        if (line[0] == '>') {  // 读长的名字
            reads.push_back(read);
            read.name = line;
            read.data = "";
            continue;
        }
        read.data += line;
    }
    reads.push_back(read);
    file.close();
    std::sort(reads.begin(), reads.end(), [](Read &a, Read &b) {
        return a.data.size() > b.data.size();  // 从大到小
    });  // 排序
    std::cout << "序列数：\t" << reads.size() << std::endl;
}
// copyData 拷贝数据
void copyData(std::vector<Read> &reads, Data &data) {
    int readsCount = reads.size();
    data.readsCount = readsCount;
    cudaMallocManaged(&data.lengths, readsCount*sizeof(int));
    cudaMallocManaged(&data.offsets, (readsCount+1)*sizeof(long));
    data.offsets[0] = 0;
    for (int i=0; i<readsCount; i++) {  // 填充lengths和offsets
        int length = reads[i].data.size();
        data.lengths[i] = length;
        data.offsets[i+1] = data.offsets[i] + length/16*16+16;
    }
    cudaMallocManaged(&data.reads, data.offsets[readsCount]*sizeof(char));
    for (int i=0; i<readsCount; i++) {  // 填充reads
        int offset = data.offsets[i];
        int length = data.lengths[i];
        memcpy(&data.reads[offset], reads[i].data.c_str(), length*sizeof(char));
    }
    cudaDeviceSynchronize();  // 同步数据
}
// kernel_baseToNumber 碱基转换为数字
void __global__ kernel_baseToNumber(char *reads, long length) {
    long index = threadIdx.x+blockDim.x*blockIdx.x;
    while (index < length) {
        switch (reads[index]) {  // 实际是寄存器计算，比用数组更快
        case 'A':
            reads[index] = 0;
            break;
        case 'a':
            reads[index] = 0;
            break;
        case 'C':
            reads[index] = 1;
            break;
        case 'c':
            reads[index] = 1;
            break;
        case 'G':
            reads[index] = 2;
            break;
        case 'g':
            reads[index] = 2;
            break;
        case 'T':
            reads[index] = 3;
            break;
        case 't':
            reads[index] = 3;
            break;
        case 'U':
            reads[index] = 3;
            break;
        case 'u':
            reads[index] = 3;
            break;
        default:
            reads[index] = 4;
            break;
        }
        index += 128*128;
    }
}
// baseToNumber 碱基转换为数字
void baseToNumber(Data &data) {
    int readsCount = data.readsCount;
    long length = data.offsets[readsCount];  // 总长度
    kernel_baseToNumber<<<128, 128>>>(data.reads, length);
    cudaDeviceSynchronize();  // 同步数据
}
// 压缩后每个碱基占两个bit，去掉gap
// kernel_compressedData 压缩数据
void __global__ kernel_compressData(int *lengths, long *offsets, char *reads,
unsigned int *compressed, int *gaps, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    long mark = offsets[index]/16;  // 压缩数据写入的位置
    int round = 0;  // round满16就把数据写入
    int gapCount = 0;  // gap的个数
    unsigned int compressedTemp = 0;  // 压缩后的数据
    long start = offsets[index];
    long end = start + lengths[index];
    for (long i=start; i<end; i++) {
        unsigned char base = reads[i];  // 读碱基
        if (base < 4) {  // 改成无判断性能没提升
            compressedTemp += base << (15-round)*2;
            round++;
            if (round == 16) {
                compressed[mark] = compressedTemp;
                compressedTemp = 0;
                round = 0;
                mark++;
            }
        } else {  // 非ATGC的碱基
            gapCount++;
        }
    }
    compressed[mark] = compressedTemp;
    gaps[index] = gapCount;
}
// compressData 压缩数据
void compressData(Data &data) {
    int readsCount = data.readsCount;
    long length = data.offsets[readsCount];
    cudaMallocManaged(&data.compressed, length/16*sizeof(unsigned int));
    cudaMallocManaged(&data.gaps, readsCount*sizeof(int));
    kernel_compressData<<<(readsCount+127)/128, 128>>> (data.lengths,
    data.offsets, data.reads, data.compressed, data.gaps, readsCount);
    cudaDeviceSynchronize();  // 同步
}
// kernel_createIndex 生成index4
void __global__ kernel_createIndex4(char *reads, int *lengths, long *offsets,
unsigned short *indexs, unsigned short *orders, long *words, int *magicBase,
int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    int start = offsets[index];
    int end = start + lengths[index];
    int magic0=0, magic1=0, magic2=0, magic3=0;  // 魔法碱基
    char bases[4];  // 实际是寄存器
    for(int i=0; i<4; i++) {  // 初始化为N
        bases[i] = 4;
    }
    int wordCount = 0;
    for (int i=start; i<end; i++) {
        for(int j=0; j<3; j++) {  // 逐步把碱基拷贝进数组
            bases[j] = bases[j+1];
        }
        bases[3] = reads[i];
        switch (bases[3]) {  // 更新magic
            case 0:
                magic0++;
                break;
            case 1:
                magic1++;
                break;
            case 2:
                magic2++;
                break;
            case 3:
                magic3++;
                break;
        }
        unsigned short indexValue = 0;
        int flag = 0;  // 是否有N
        for (int j=0; j<4; j++) {
            indexValue += (bases[j]&3)<<(3-j)*2;
            flag += max(bases[j]-3, 0);
        }
        indexs[i] = flag?65535:indexValue;  // 遇到N就存入最大值
        wordCount += flag?0:1;
    }
    words[index] = wordCount;  // index长度
    magicBase[index*4+0] = magic0;  // 更新magicBase
    magicBase[index*4+1] = magic1;
    magicBase[index*4+2] = magic2;
    magicBase[index*4+3] = magic3;
}
// kernel_createIndex 生成index5
void __global__ kernel_createIndex5(char *reads, int *lengths, long *offsets,
unsigned short *indexs, unsigned short *orders, long *words, int *magicBase,
int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    int start = offsets[index];
    int end = start + lengths[index];
    int magic0=0, magic1=0, magic2=0, magic3=0;  // 魔法碱基
    char bases[5];  // 实际是寄存器
    for(int i=0; i<5; i++) {  // 初始化为N
        bases[i] = 4;
    }
    int wordCount = 0;
    for (int i=start; i<end; i++) {
        for(int j=0; j<4; j++) {  // 逐步把碱基拷贝进数组
            bases[j] = bases[j+1];
        }
        bases[4] = reads[i];
        switch (bases[4]) {  // 更新magic
            case 0:
                magic0++;
                break;
            case 1:
                magic1++;
                break;
            case 2:
                magic2++;
                break;
            case 3:
                magic3++;
                break;
        }
        unsigned short indexValue = 0;
        int flag = 0;  // 是否有N
        for (int j=0; j<5; j++) {
            indexValue += (bases[j]&3)<<(4-j)*2;
            flag += max(bases[j]-3, 0);
        }
        indexs[i] = flag?65535:indexValue;  // 遇到N就存入最大值
        wordCount += flag?0:1;
    }
    words[index] = wordCount;  // index长度
    magicBase[index*4+0] = magic0;  // 更新magicBase
    magicBase[index*4+1] = magic1;
    magicBase[index*4+2] = magic2;
    magicBase[index*4+3] = magic3;
}
// kernel_createIndex 生成index6
void __global__ kernel_createIndex6(char *reads, int *lengths, long *offsets,
unsigned short *indexs, unsigned short *orders, long *words, int *magicBase,
int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    int start = offsets[index];
    int end = start + lengths[index];
    int magic0=0, magic1=0, magic2=0, magic3=0;  // 魔法碱基
    char bases[6];  // 实际是寄存器
    for(int i=0; i<6; i++) {  // 初始化为N
        bases[i] = 4;
    }
    int wordCount = 0;
    for (int i=start; i<end; i++) {
        for(int j=0; j<5; j++) {  // 逐步把碱基拷贝进数组
            bases[j] = bases[j+1];
        }
        bases[5] = reads[i];
        switch (bases[5]) {  // 更新magic
            case 0:
                magic0++;
                break;
            case 1:
                magic1++;
                break;
            case 2:
                magic2++;
                break;
            case 3:
                magic3++;
                break;
        }
        unsigned short indexValue = 0;
        int flag = 0;  // 是否有N
        for (int j=0; j<6; j++) {
            indexValue += (bases[j]&3)<<(5-j)*2;
            flag += max(bases[j]-3, 0);
        }
        indexs[i] = flag?65535:indexValue;  // 遇到N就存入最大值
        wordCount += flag?0:1;
    }
    words[index] = wordCount;  // 先存index长度，后面会改为offset
    magicBase[index*4+0] = magic0;  // 更新magicBase
    magicBase[index*4+1] = magic1;
    magicBase[index*4+2] = magic2;
    magicBase[index*4+3] = magic3;
}
// kernel_createIndex 生成index7
void __global__ kernel_createIndex7(char *reads, int *lengths, long *offsets,
unsigned short *indexs, unsigned short *orders, long *words, int *magicBase,
int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    int start = offsets[index];
    int end = start + lengths[index];
    int magic0=0, magic1=0, magic2=0, magic3=0;  // 魔法碱基
    char bases[7];  // 实际是寄存器
    for(int i=0; i<7; i++) {  // 初始化为N
        bases[i] = 4;
    }
    int wordCount = 0;
    for (int i=start; i<end; i++) {
        for(int j=0; j<6; j++) {  // 逐步把碱基拷贝进数组
            bases[j] = bases[j+1];
        }
        bases[6] = reads[i];
        switch (bases[6]) {  // 更新magic
            case 0:
                magic0++;
                break;
            case 1:
                magic1++;
                break;
            case 2:
                magic2++;
                break;
            case 3:
                magic3++;
                break;
        }
        unsigned short indexValue = 0;
        int flag = 0;  // 是否有N
        for (int j=0; j<7; j++) {
            indexValue += (bases[j]&3)<<(6-j)*2;
            flag += max(bases[j]-3, 0);
        }
        indexs[i] = flag?65535:indexValue;  // 遇到N就存入最大值
        wordCount += flag?0:1;
    }
    words[index] = wordCount;  // index长度
    magicBase[index*4+0] = magic0;  // 更新magicBase
    magicBase[index*4+1] = magic1;
    magicBase[index*4+2] = magic2;
    magicBase[index*4+3] = magic3;
}
// createIndex 生成index
void createIndex(Data &data, Option &option) {
    int readsCount = data.readsCount;
    int wordLength = option.wordLength;
    int length = data.offsets[readsCount];
    cudaMallocManaged(&data.indexs, length*sizeof(unsigned short));  // index值
    cudaMallocManaged(&data.orders, length*sizeof(unsigned short));  // index秩
    cudaMallocManaged(&data.words, (readsCount+1)*sizeof(long));  // index长度
    cudaMallocManaged(&data.magicBase, readsCount*4*sizeof(int));  // 魔法碱基
    switch (wordLength) {
        case 4:
            kernel_createIndex4<<<(readsCount+127)/128, 128>>>
            (data.reads, data.lengths, data.offsets,
            data.indexs, data.orders, data.words, data.magicBase, readsCount);
            break;
        case 5:
            kernel_createIndex5<<<(readsCount+127)/128, 128>>>
            (data.reads, data.lengths, data.offsets,
            data.indexs, data.orders, data.words, data.magicBase, readsCount);
            break;
        case 6:
            kernel_createIndex6<<<(readsCount+127)/128, 128>>>
            (data.reads, data.lengths, data.offsets,
            data.indexs, data.orders, data.words, data.magicBase, readsCount);
            break;
        case 7:
            kernel_createIndex7<<<(readsCount+127)/128, 128>>>
            (data.reads, data.lengths, data.offsets,
            data.indexs, data.orders, data.words, data.magicBase, readsCount);
            break;
    }
    cudaDeviceSynchronize();  // 同步数据
}
// kernel_createCutoff 生成阈值
void __global__ kernel_createCutoff(float threshold, int wordLength,
int *lengths, long *words, int *wordCutoff, int readsCount) {
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= readsCount) return;  // 超出范围
    int length = lengths[index];
    int required = length - wordLength + 1;
    int cutoff = ceil((float)length*(1.0-threshold)*(float)wordLength);
    required -= cutoff;
    wordCutoff[index] = required;
}
// createCutoff 生成阈值
void createCutoff(Data &data, Option &option) {
    int readsCount = data.readsCount;
    cudaMallocManaged(&data.wordCutoff, readsCount*sizeof(int));  // word阈值
    kernel_createCutoff<<<(readsCount+127)/128, 128>>>
    (option.threshold, option.wordLength, data.lengths,
    data.words, data.wordCutoff, readsCount);
    cudaDeviceSynchronize();  // 同步数据
}
// sortIndex 排序index
void sortIndex(Data &data) {
    int readsCount = data.readsCount;
    for (int i=0; i<readsCount; i++) {
        int start = data.offsets[i];
        int length = data.words[i];
        std::sort(&data.indexs[start], &data.indexs[start]+length);
    }
    cudaDeviceSynchronize();  // 同步数据
}
// kernel_mergeIndex 合并相同index
void __global__ kernel_mergeIndex(long *offsets, unsigned short *indexs,
unsigned short *orders, long *words, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    int start = offsets[index];
    int end = start + words[index];
    unsigned short basePrevious = indexs[start];
    unsigned short baseNow;
    int count = 1;
    for (int i=start+1; i<end; i++) {  // 合并相同的index，orders为相同个数
        baseNow = indexs[i];
        if (baseNow == basePrevious) {
            count++;
            orders[i-1] = 0;
        } else {
            basePrevious = baseNow;
            orders[i-1] = count;
            count = 1;
        }
    }
    orders[end-1] = count;
}
// mergeIndex 合并相同index
void mergeIndex(Data &data) {
    int readsCount = data.readsCount;
    kernel_mergeIndex<<<(readsCount+127)/128, 128>>>
    (data.offsets, data.indexs, data.orders, data.words, readsCount);
    cudaDeviceSynchronize();  // 同步数据
}
// updateRepresentative 更新代表序列
void updateRepresentative(int *cluster, int &representative, int readsCount) {
    representative++;
    while (representative < readsCount) {
        if (cluster[representative] < 0) {  // 遇到代表序列了
            cluster[representative] = representative;
            break;
        }
        representative++;
    }
}
// kernel_makeTable 生成table
void __global__ kernel_makeTable(long *offsets, unsigned short *indexs,
unsigned short *orders, long *words,
unsigned short *table, int representative) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    int start = offsets[representative];
    int end = start + words[representative];
    for (int i=index+start; i<end; i+=128*128) {  // 写标记
        unsigned short order = orders[i];
        if (order == 0) continue;  // 非标记位就跳过
        table[indexs[i]] = order;
    }
}
// kernel_cleanTable 清零
void __global__ kernel_cleanTable(long *offsets, unsigned short *indexs,
unsigned short *orders,  long *words,
unsigned short *table, int representative) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    int start = offsets[representative];
    int end = start + words[representative];
    for (int i=index+start; i<end; i+=128*128) {  // 写标记
        unsigned short order = orders[i];
        if (order == 0) continue;  // 非标记位就跳过
        table[indexs[i]] = 0;
    }
}
// kernel_magic 魔法过滤
void __global__ kernel_magic(float threshold, int *lengths, int *magicBase,
int *cluster, int representative, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    if (cluster[index] >= 0) return;  // 已经聚过类了
    int offsetOne = representative*4;  // 代表序列magic的位置
    int offsetTwo = index*4;  // 查询序列magic的位置
    int magic = min(magicBase[offsetOne+0], magicBase[offsetTwo+0]) +
    min(magicBase[offsetOne+1], magicBase[offsetTwo+1]) +
    min(magicBase[offsetOne+2], magicBase[offsetTwo+2]) +
    min(magicBase[offsetOne+3], magicBase[offsetTwo+3]);
    int length = lengths[index];
    int minLength = ceil((float)length*threshold);
    if (magic > minLength) {  // 超过阈值就进行下一步
        cluster[index] = -2;
    }
}
// kernel_filter 过滤
void __global__ kernel_filter(float threshold, int wordLength, int *lengths,
long *offsets, unsigned short *indexs, unsigned short *orders, long *words,
int *wordCutoff, int *cluster, unsigned short *table, int readsCount) {
    if (blockIdx.x >= readsCount) return;  // 超出范围
    if (cluster[blockIdx.x] != -2) return;  // 没通过魔法过滤
    __shared__ int result[128];  // 每个线程的结果
    result[threadIdx.x] = 0;  // 清零
    int start = offsets[blockIdx.x];
    int end = start + words[blockIdx.x];
    for (int i=threadIdx.x+start; i<end; i+=128) {  // 比对标记
        result[threadIdx.x] += min(table[indexs[i]], orders[i]);
    }
    __syncthreads();  // 同步一下
    for (int i=128/2; i>0; i/=2) {  // 规约
        if (threadIdx.x>=i) return;  // 超出范围
        result[threadIdx.x] += result[threadIdx.x+i];
        __syncthreads();
    }
    if(result[0] > wordCutoff[blockIdx.x]) {  // 超过阈值就交给动态规划
        cluster[blockIdx.x] = -3;
    } else {
        cluster[blockIdx.x] = -1;  // 恢复原值
    }
}
// kernel_align 动态规划
void __global__ kernel_align(float threshold, int *lengths, long *offsets, 
unsigned int *compressed, int *gaps, int representative,
int *cluster, int readsCount, int bufLength) {
    // 定义变量
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    if (cluster[index] != -3) return;  // 没通过常规过滤
    int target = representative;  // 代表读长
    int query = index;  // 比对读长
    int minLength = ceil((float)lengths[index]*threshold);
    int targetLength = lengths[target] - gaps[target];  // 代表读长碱基数
    int queryLength = lengths[query] - gaps[query];  // 比对读长碱基数
    int target32Length = targetLength/16+1;  // target的32bit长度
    int query32Length  = queryLength/16+1;  // query的32bit长度
    int targetOffset = offsets[target]/16;  // 代表读长偏移
    int queryOffset = offsets[query]/16;  // 比对读长偏移
    short rowNow[65536] = {0};  // 动态规划矩阵
    short rowPrevious[65536] = {0};  // 动态规划矩阵
    int columnPrevious[17] = {0};  // 前一列 用共享内存反而更慢
    int columnNow[17] = {0};  // 当前列 用寄存器和全局数组速度相同
    int shift = ceil((float)targetLength-(float)queryLength*threshold);
    shift = ceil((float)shift/16.0);  // 相对对角线，左右偏移方块个数
    // 开始计算
    for (int i = 0; i < query32Length; i++) {  // query是列
        // 第一次大循环
        for (int j=0; j<17; j++) {  // 清空两个比清空一个更快
            columnPrevious[j] = 0;
            columnNow[j] = 0;
        }
        int targetIndex = 0;  // target已比对的碱基数
        unsigned int queryPack = compressed[queryOffset+i];  // 加载16个query碱基
        int jstart = i-shift;
        jstart = max(jstart, 0);
        int jend = i+shift;
        jend = min(jend, target32Length);
        for (int j=0; j<target32Length; j++) {  // target是行
            columnPrevious[0] = rowPrevious[targetIndex];
            unsigned int targetPack = compressed[targetOffset+j];  // 加载16个target碱基
            //---16*16核心----(32位寄存器最快，用64位反而慢一倍多，16位也慢)
            for (int k=30; k>=0; k-=2) {  // 循环16次 依次取target碱基
                // 第一次小循环
                int targetBase = (targetPack>>k)&3;  // 从target取一个碱基
                int m=0;
                columnNow[m] = rowPrevious[targetIndex+1];
                for (int l=30; l>=0; l-=2) {  // 循环16次 依次取query碱基
                    m++;
                    int queryBase = (queryPack>>l)&3;  // 从query取一个碱基
                    int diffScore = queryBase == targetBase;
                    columnNow[m] = columnPrevious[m-1] + diffScore;
                    columnNow[m] = max(columnNow[m], columnNow[m-1]);
                    columnNow[m] = max(columnNow[m], columnPrevious[m]);
                }
                targetIndex++;
                rowNow[targetIndex] = columnNow[16];
                if (targetIndex == targetLength) {  // 动态规划矩阵最后一列
                    if(i == query32Length-1) {  // 比对完成
                        int score = columnNow[queryLength%16];
                        if (score >= minLength) {
                            cluster[index] = target;
                        } else {
                            cluster[index] = -1;
                        }
                        return;
                    }
                    break;
                }
                // 第二次小循环 columnPrevious与columnNow调换位置
                k-=2;
                targetBase = (targetPack>>k)&3;  // 从target取一个碱基
                m=0;
                columnPrevious[m] = rowPrevious[targetIndex+1];
                for (int l=30; l>=0; l-=2) {  // 循环16次 依次取query碱基
                    m++;
                    int queryBase = (queryPack>>l)&3;  // 从query取一个碱基
                    int diffScore = queryBase == targetBase;
                    columnPrevious[m] = columnNow[m-1] + diffScore;
                    columnPrevious[m] = max(columnPrevious[m], columnPrevious[m-1]);
                    columnPrevious[m] = max(columnPrevious[m], columnNow[m]);
                }
                targetIndex++;
                rowNow[targetIndex] = columnPrevious[16];
                if (targetIndex == targetLength) {  // 动态规划矩阵最后一列
                    if(i == query32Length-1) {  // 比对完成
                        int score = columnPrevious[queryLength%16];
                        if (score >= minLength) {
                            cluster[index] = target;
                        } else {
                            cluster[index] = -1;
                        }
                        return;
                    }
                    break;
                }
            }
        }
        // 第二次大循环 rowPrevious与rowNow调换位置
        i++;
        for (int j=0; j<17; j++) {  // 清空两个比清空一个更快
            columnPrevious[j] = 0;
            columnNow[j] = 0;
        }
        targetIndex = 0;  // target已比对的碱基数
        queryPack = compressed[queryOffset+i];  // 加载16个query碱基
        jstart = i-shift;
        jstart = max(jstart, 0);
        jend = i+shift;
        jend = min(jend, target32Length);
        for (int j=0; j<target32Length; j++) {  // target是行
            unsigned int targetPack = compressed[targetOffset+j];  // 加载16个target碱基
            //---16*16核心----(32位寄存器最快，用64位反而慢一倍多)
            for (int k=30; k>=0; k-=2) {  // 循环16次 依次取target碱基
                // 第一次小循环
                int targetBase = (targetPack>>k)&3;  // 从target取一个碱基
                int m=0;
                columnNow[m] = rowNow[targetIndex+1];
                for (int l=30; l>=0; l-=2) {  // 循环16次 依次取query碱基
                    m++;
                    int queryBase = (queryPack>>l)&3;  // 从query取一个碱基
                    int diffScore = queryBase == targetBase;
                    columnNow[m] = columnPrevious[m-1] + diffScore;
                    columnNow[m] = max(columnNow[m], columnNow[m-1]);
                    columnNow[m] = max(columnNow[m], columnPrevious[m]);
                }
                targetIndex++;
                rowPrevious[targetIndex] = columnNow[16];
                if (targetIndex == targetLength) {  // 动态规划矩阵最后一列
                    if(i == query32Length-1) {  // 比对完成
                        int score = columnNow[queryLength%16];
                        if (score >= minLength) {
                            cluster[index] = target;
                        } else {
                            cluster[index] = -1;
                        }
                        return;
                    }
                    break;
                }
                // 第二次小循环 columnPrevious与columnNow调换位置
                k-=2;
                targetBase = (targetPack>>k)&3;  // 从target取一个碱基
                m=0;
                columnPrevious[m] = rowNow[targetIndex+1];
                for (int l=30; l>=0; l-=2) {  // 循环16次 依次取query碱基
                    m++;
                    int queryBase = (queryPack>>l)&3;  // 从query取一个碱基
                    int diffScore = queryBase == targetBase;
                    columnPrevious[m] = columnNow[m-1] + diffScore;
                    columnPrevious[m] = max(columnPrevious[m], columnPrevious[m-1]);
                    columnPrevious[m] = max(columnPrevious[m], columnNow[m]);
                }
                targetIndex++;
                rowPrevious[targetIndex] = columnPrevious[16];
                if (targetIndex == targetLength) {  // 动态规划矩阵最后一列
                    if(i == query32Length-1) {  // 比对完成
                        int score = columnPrevious[queryLength%16];
                        if (score >= minLength) {
                            cluster[index] = target;
                        } else {
                            cluster[index] = -1;
                        }
                        return;
                    }
                    break;
                }
            }
        }
    }
}
// 聚类
void clustering(Option &option, Data &data, Bench &bench) {
    int readsCount = data.readsCount;
    cudaMallocManaged(&bench.cluster, readsCount*sizeof(int));  // 聚类结果
    for (int i=0; i<readsCount; i++) {
        bench.cluster[i] = -1;
    }
    cudaMallocManaged(&bench.table, 65536*sizeof(unsigned short));  // table
    memset(bench.table, 0, 65536*sizeof(unsigned short));  // 清0
    bench.representative = -1;
    std::cout << "代表序列/总序列数:" << std::endl;
    while (bench.representative < readsCount) {  // 聚类
cudaDeviceSynchronize();  // 同步数据
        updateRepresentative
        (bench.cluster, bench.representative, readsCount);  // 更新代表序列
        std::cout << bench.representative << "/" << readsCount << std::endl;
cudaDeviceSynchronize();  // 同步数据
        kernel_makeTable<<<128, 128>>>
        (data.offsets, data.indexs, data.orders, data.words,
        bench.table, bench.representative);  // 生成table
cudaDeviceSynchronize();  // 同步数据
        kernel_magic<<<(readsCount+127)/128, 128>>>
        (option.threshold, data.lengths, data.magicBase,
        bench.cluster, bench.representative,readsCount);  // 魔法过滤
cudaDeviceSynchronize();  // 同步数据
        kernel_filter<<<readsCount, 128>>>
        (option.threshold, option.wordLength, data.lengths,
        data.offsets, data.indexs, data.orders, data.words,
        data.wordCutoff, bench.cluster, bench.table, readsCount);  // word过滤
cudaDeviceSynchronize();  // 同步数据
        kernel_align<<<(readsCount+127)/128, 128>>>
        (option.threshold, data.lengths, data.offsets, data.compressed,
        data.gaps, bench.representative, bench.cluster, readsCount, 3000);  // 比对
cudaDeviceSynchronize();  // 同步数据
        kernel_cleanTable<<<128, 128>>>
        (data.offsets, data.indexs, data.orders, data.words,
        bench.table, bench.representative);  // 清零table
        cudaDeviceSynchronize();  // 同步数据
        // std::cout << std::endl;
    }
}
// saveFile 保存结果
void saveFile(Option &option, std::vector<Read> &reads, Bench &bench) {
    std::ofstream file(option.outputFile);
    int sum = 0;
    for (int i=0; i<reads.size(); i++) {
        if (bench.cluster[i] == i) {
            file << reads[i].name << std::endl;
            file << reads[i].data << std::endl;
            sum++;
        }
    }
    file.close();
    std::cout << "聚类：" << sum << "个" << std::endl;
}
// checkValue 查看运行的结果是否正确
void checkValue(Data &data) {
}
// 检查显卡错误
void checkError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    }
}