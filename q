// cluster.cpp
// 一个聚类软件
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include<CL/sycl.hpp>
using namespace cl::sycl;

#define MAXREADLENGTH 2<<16  //读长的长度，65536
#define THREADNUMBER 128  // 每个块中线程数 目前也是序列的最大长度限制，不能改了。
#define KMERLENGTH 8  // index table中k-mer的长度
#define SIMILARITYTHRESHOLD 0.90  // 相似度的阈值

// Read是记录读长的数据结构
struct Read {  // 一条记录，称为一条read
    int index;  // read名，实际就是read在文件中的序号
    unsigned char *readData;  // read的实际内容
    int readDataLength;  // read的长度
    int *indexTable;  // index table
    int cluster;  // 属于哪一个类，实际就是clusters数组的序号
};

Read *readFile(char *fileName, int *readsLength) {
    clock_t start, stop;
    double cost;
    start = clock();
    // 打开文件
    FILE *fastaFile = NULL;
    fastaFile = fopen(fileName, "r");
    if(fastaFile == NULL) {
        printf("打开%s文件失败\n", fileName);
        return NULL;
    }
    // 开始读文件
    Read *reads;  // Read数组，最后要返回的值
    *readsLength = 1024;  // 初始分配的reads序列的长度，1024，如果不够就默认翻倍
    reads = (Read*)malloc(*readsLength * sizeof(Read));
    char readBuffer[MAXREADLENGTH];  // 默认一行的长度不超过65536个字符
    int readBufferLength = 0;  // 读到了多长的buffer
    int readLines = 0;  // 现在读到了多少行read
    while(true) {
        readBuffer[0] = '\0';  // '\0'表示字符串结束，类似清空字符串的效果
        fgets(readBuffer, MAXREADLENGTH, fastaFile);
        readBufferLength = strlen(readBuffer);
        if(readBufferLength == 0) {  // 没读到内容就是文件读完了，打破循环
            break;
        }
        if(readBuffer[0] == '>') {  // 如果开头是'>'就是注释行，跳过
            continue;
        }
        if(readLines == *readsLength) {  // reads的长度不够，则翻倍
            *readsLength *= 2;
            reads = (Read*)realloc(reads, *readsLength * sizeof(Read));
        }
        reads[readLines].index = readLines;
        reads[readLines].readDataLength = readBufferLength-1;
        reads[readLines].readData = (unsigned char*)malloc(reads[readLines].readDataLength * sizeof(unsigned char));
        for(int i=0; i<reads[readLines].readDataLength; i++) {  // 把字符转为数字，方便后期计算，后期调优可以cuda一下
            switch(readBuffer[i]) {
                case 'A':
                reads[readLines].readData[i] = 0;
                break;
                case 'C':
                reads[readLines].readData[i] = 1;
                break;
                case 'G':
                reads[readLines].readData[i] = 2;
                break;
                case 'T':
                reads[readLines].readData[i] = 3;
                break;
                default:
                reads[readLines].readData[i] = 4;
                break;
            }
        }
        readLines += 1;
    }
    
    *readsLength = readLines;
    reads = (Read*)realloc(reads, *readsLength * sizeof(Read));
    // 收尾工作
    fclose(fastaFile);
    stop = clock();
    cost = (double)(stop-start)/CLOCKS_PER_SEC;
    printf("读取%s文件完成，耗时%fs\n", fileName, cost);
    return reads;
}

void sortReadsOrder(Read *reads, int left, int right) {
    if(left > right) {
        return;
    }
    int i = left;
    int j = right;
    Read temp = reads[i];
    while(i < j) {
        while((reads[j].readDataLength<=temp.readDataLength)&&(i<j)) {
            j--;
        }
        reads[i] = reads[j];
        while((reads[i].readDataLength>=temp.readDataLength)&&(i<j)) {
            i++;
        }
        reads[j] = reads[i];
    }
    reads[i] = temp;
    sortReadsOrder(reads, left, i-1);
    sortReadsOrder(reads, j+1, right);
}

// getIndexTable 生成index table，结果放在reads中。
void getIndexTable(Read *reads, int readsLength) {
    clock_t start, stop;
    double cost;
    start = clock();
    
    for(int i=0; i<readsLength; i++) {
        // 定义数据结构
        default_selector my_selector;
        queue my_queue(my_selector);
        reads[i].indexTable = (int*)malloc((1<<KMERLENGTH*2)*sizeof(int));  //index table 数据结构
        memset(reads[i].indexTable , 0, (1<<KMERLENGTH*2)*sizeof(int));  // index table 清零
        buffer<unsigned char, 1> readData_buffer(reads[i].readData, range<1>(reads[i].readDataLength));
        buffer<int, 1> readIndexTable_buffer(reads[i].indexTable, range<1>(1<<KMERLENGTH*2));
        // 计算index table的核函数
        my_queue.submit([&](handler &my_handler) {
            stream cout(1024, 256, my_handler);  // 打印log
            auto readData_accessor = readData_buffer.get_access<access::mode::read>(my_handler);
            auto readIndexTable_accessor = readIndexTable_buffer.get_access<access::mode::read_write>(my_handler);
            my_handler.parallel_for<class test>(range<1>(reads[i].readDataLength-KMERLENGTH+1), [=](id<1> index) {
            // 结尾的几个元素没法算，所以-KMERLENGTH+1
                int indexTableIndex = 0;  // index table的索引
                int indexTableValue = 1;  // index table的索引位置要增加的值
                for(int j=0; j<KMERLENGTH; j++) {
                    if(readData_accessor[index+i] != 4) {
                        indexTableIndex += (int)readData_accessor[index+j]*(1<<j*2);
                    } else {  // 如果是4，说明这是个未知碱基，直接退出
                        indexTableValue = 0;
                        break;
                    }
                }
                atomic<int> readIndexTable_counter{global_ptr<int> {&readIndexTable_accessor[indexTableIndex]}};
                readIndexTable_counter.fetch_add(indexTableValue);
            });
        });
    }
    stop = clock();
    cost = (double)(stop-start)/CLOCKS_PER_SEC;
    printf("生成indexTable完成，耗时%fs\n", cost);
}

int indexTableFilter(Read readOne, Read readTwo) {
    // 定义变量
    int sameWords;
    buffer<int> sameWords_buffer(&sameWords, 1);
    buffer<int, 1> indexTableOne_buffer(readOne.indexTable, range<1>(1<<KMERLENGTH*2));
    buffer<int, 1> indexTableTwo_buffer(readTwo.indexTable, range<1>(1<<KMERLENGTH*2));
    default_selector my_selector;
    queue my_queue(my_selector);
    // 计算index table的same words的核函数
    my_queue.submit([&](handler &my_handler) {
        auto sameWords_accessor = sameWords_buffer.get_access<access::mode::read_write>(my_handler);
        auto indexTableOne_accessor = indexTableOne_buffer.get_access<access::mode::read>(my_handler);
        auto indexTableTwo_accessor = indexTableTwo_buffer.get_access<access::mode::read>(my_handler);
        my_handler.parallel_for<class test>(range<1>(1<<KMERLENGTH*2), [=](id<1> index) {
            if(indexTableOne_accessor[index]>0 or indexTableTwo_accessor[index]>0) {
                atomic<int> sameWords_atomic{global_ptr<int> {&sameWords_accessor[0]}};
                sameWords_atomic.fetch_add(min(indexTableOne_accessor[index], indexTableTwo_accessor[index]));
            }
        });
    });
    //收尾
    my_queue.wait_and_throw();
    sameWords_buffer.get_access<access::mode::read>();
    return sameWords<(readTwo.readDataLength*(1-(1-SIMILARITYTHRESHOLD)*KMERLENGTH)-KMERLENGTH+1)?0:1; 
}

// alignReads 输入两个read和一个阈值，返回一个int，用动态规划的方法计算两个read的相似度，1表示匹配度超阈值，0表示没超
int alignReads(Read readOne, Read readTwo) {
    // 定义变量
    int subsequenceLength=0;  // 最长公共子串长度
    buffer<unsigned char, 1> dataOne_buffer(readOne.readData, range<1>(readOne.readDataLength));
    buffer<unsigned char, 1> dataTwo_buffer(readTwo.readData, range<1>(readTwo.readDataLength));
    buffer<int> subsequenceLength_buffer(&subsequenceLength, 1);
    buffer<int> readOneLength_buffer(&readOne.readDataLength, 1);
    buffer<int> readTwoLength_buffer(&readTwo.readDataLength, 1);
    default_selector my_selector;
    queue my_queue(my_selector);
    // 计算index table的same words的核函数
    my_queue.submit([&](handler &my_handler) {
        auto dataOne_accessor = dataOne_buffer.get_access<access::mode::read>(my_handler);
        auto dataTwo_accessor = dataTwo_buffer.get_access<access::mode::read>(my_handler);
        auto subsequenceLength_accessor = subsequenceLength_buffer.get_access<access::mode::read_write>(my_handler);
        auto readOneLength_accessor = readOneLength_buffer.get_access<access::mode::read_write>(my_handler);
        auto readTwoLength_accessor = readTwoLength_buffer.get_access<access::mode::read_write>(my_handler);
        accessor<int, 1, access::mode::read_write, access::target::local> tempPrevious_mem(range<1>(readTwo.readDataLength+1), my_handler);
        accessor<int, 1, access::mode::read_write, access::target::local> tempNow_mem(range<1>(readTwo.readDataLength+1), my_handler);
        accessor<int, 1, access::mode::read_write, access::target::local> tempNext_mem(range<1>(readTwo.readDataLength+1), my_handler);
        stream cout(1024, 256, my_handler);  // 打印log
        my_handler.parallel_for<class test>(nd_range<1>(256, 256), [=](nd_item<1> item) {
            // 清零temp
            int index = item.get_local_linear_id();
            while(index < readTwoLength_accessor[0]+1) {
                tempPrevious_mem[index] = 0;
                tempNow_mem[index] = 0;
                tempNext_mem[index] = 0;
                index += 256;
            }
            item.barrier(access::fence_space::local_space);  // 保证清零完成
            // 计算动态规划矩阵
            index = item.get_local_linear_id()+1;
            for(int i=1; i<readOneLength_accessor[0]+readTwoLength_accessor[0]; i++) {
                index = item.get_local_linear_id()+1;
                while(index < readTwoLength_accessor[0]+1) {  // 计算tempNext
                    if(index-1<i && i<readOneLength_accessor[0]+index) {
                        if(dataOne_accessor[i-index] == dataTwo_accessor[index-1]) {
                            tempNext_mem[index] = tempPrevious_mem[index-1]+1;
                        }
                        tempNext_mem[index] = max(tempNext_mem[index], tempPrevious_mem[index-1]);
                        tempNext_mem[index] = max(tempNext_mem[index], tempNow_mem[index]);
                        tempNext_mem[index] = max(tempNext_mem[index], tempNow_mem[index-1]);
                    }
                    index += 256;
                }
                item.barrier(access::fence_space::local_space);  // 保证tempNext都计算完成
                index = item.get_local_linear_id()+1;
                while(index < readTwoLength_accessor[0]+1) {  // 滑动tempNext，tempNow，tempPrevious
                    tempPrevious_mem[index] = tempNow_mem[index];
                    tempNow_mem[index] = tempNext_mem[index];
                    index += 256;
                }
                item.barrier(access::fence_space::local_space);  // 保证滑动都完成
            }
            // 输出结果
            index = item.get_local_linear_id();
            if(index == 0) {
                subsequenceLength_accessor[0] = tempNow_mem[readTwoLength_accessor[0]];
            }
        });
    });
    //收尾
    my_queue.wait_and_throw();
    subsequenceLength_buffer.get_access<access::mode::read>();
    return subsequenceLength<readTwo.readDataLength*SIMILARITYTHRESHOLD?0:1;
}

// clustering输入Reads数组，调用cuda计算相似度，返回cluster数组
Read *clustering(Read *reads, int readsLength, int *clustersLength) {
    // 定义变量
    clock_t start, stop;
    double cost;
    start = clock();
    Read *clusters;
    *clustersLength = 10;  // 聚类的初始长度设为10，长度每次倍增
    clusters = (Read*)malloc(*clustersLength*sizeof(Read));
    int clustersTop = 0;  // 把聚类看作一个栈，这个就是栈顶
    // 遍历reads与clusters比对，生成clusters
    for(int i=0; i<readsLength; i++) {  // i遍历reads
        int belongToCluster = 0;  // 1表示聚到某个类，0表示没聚到某个类
        for(int j=0; j<clustersTop; j++) {  // j遍历clusters
            if(indexTableFilter(clusters[j], reads[i])) {  // 如果没被过滤掉，就接着比对
                if(alignReads(clusters[j], reads[i])) {  // 如果比对后聚类成功，就加入新类
                    reads[i].cluster = j;
                    belongToCluster = 1;  // 比对后聚类成功
                    break;
                }
            }
        }
        if(belongToCluster == 0) {  // read没有归到现有的cluster当中，则成为一个新类
            if(clustersTop == *clustersLength) {
                *clustersLength *= 2;
                clusters = (Read*)realloc(clusters, *clustersLength*sizeof(Read));
            }
            reads[i].cluster = -1;
            clusters[clustersTop] = reads[i];
            clustersTop += 1;
        }
    }
    // 收尾工作
    *clustersLength = clustersTop;
    clusters = (Read*)realloc(clusters, *clustersLength*sizeof(Read));
    stop = clock();
    cost = (double)(stop-start)/CLOCKS_PER_SEC;
    printf("聚类完成，耗时%fs\n", cost);
    return clusters;
}

int main(int argc, char *argv[]) {
    // 开发完成
    char fileName[] = "testData.fasta";
    int readsLength = 0;
    Read *reads = readFile(fileName, &readsLength);  // 测试ok 读序列
    sortReadsOrder(reads, 0, readsLength-1);  // 测试ok 序列排序
    getIndexTable(reads, readsLength);  // 测试ok,生成index table
    // 开发中
    int clustersLength = 0;
    Read *clusters = clustering(reads, readsLength, &clustersLength);  // 生成聚类结果
    printf("聚为了%d类\n", clustersLength);
    printf("done!\n");
}
