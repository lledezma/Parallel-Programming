#include <iostream>
#include "StopWatch.h"
#include <stdio.h>
#include <string.h>
#include "PatternMatch.h"
#include <fstream>
#include <omp.h>
using namespace std;

int badchar[256];
int global_count;

char* loadFile(const char* fileName, int& fsize);
void ParallelMatchNaive(const char* txt, int N, const char* pat, int M);
void ParallelMatchBoyerMoore(const char* txt, int N, const char* pat, int M);



int main() {
    const int MAXCHAR = 0;
    const char* text = loadFile("textdata.txt", MAXCHAR);

    if (text == nullptr) {
        printf("Failure at opening text file!\n");
        return 0;
    }

    int N = strlen(text);
    const int PATTERN_COUNT = 5;
    const char* pattern[PATTERN_COUNT] = { "and","fanny","removing", "possible humoured", "six hearted hundred towards" };

    int countNaive = 0;
    int countBoyer = 0;
    double eTimeNaive = 0;
    double eTimeBoyer = 0;
    StopWatch sw;

    printf("+------------------------------+---------------------+---------------------+\n");
    printf("|%-30s|%-21s|%-21s|\n", "SERIAL", "NAIVE", "BOYER_MOORE");
    printf("+------------------------------+---------------------+---------------------+\n");
    printf("|%-30s|%-10s|%-10s|%-10s|%-10s|\n", "Pattern", "Count", "Time(ms)", "Count", "Time(ms)");
    printf("+------------------------------+---------------------+---------------------+\n");
    for (int i = 0; i < PATTERN_COUNT; i++)
    {
        int M = strlen(pattern[i]);
        preprocessHeuristicArray(pattern[i], M, badchar);

        sw.start();
        countNaive = NaiveMatch(text, N, pattern[i], M);
        sw.stop();
        eTimeNaive = sw.elapsedTime();

        sw.start();
        countBoyer = BoyerMooreMatch(text, N, pattern[i], M, badchar);
        sw.stop();
        eTimeBoyer = sw.elapsedTime();

        printf("|%-30s|%-10d|%-10.4lf|%-10d|%-10.4lf|\n", pattern[i], countNaive, eTimeNaive, countBoyer, eTimeBoyer);
    }

    printf("+------------------------------+---------------------+---------------------+\n");
    printf("\n\n");

    printf("+------------------------------+---------------------+---------------------+\n");
    printf("|%-30s|%-21s|%-21s|\n", "PARALLEL", "NAIVE", "BOYER_MOORE");
    printf("+------------------------------+---------------------+---------------------+\n");
    printf("|%-30s|%-10s|%-10s|%-10s|%-10s|\n", "Pattern", "Count", "Time(ms)", "Count", "Time(ms)");
    printf("+------------------------------+---------------------+---------------------+\n");
    for (int i = 0; i < PATTERN_COUNT; i++) 
    {
        int M = strlen(pattern[i]);
        preprocessHeuristicArray(pattern[i], M, badchar);

        global_count = 0;
        sw.start();
#pragma omp parallel num_threads(4)
        ParallelMatchNaive(text, N, pattern[i], M);
        sw.stop();
        eTimeNaive = sw.elapsedTime();
        countNaive = global_count;


        global_count = 0;
        sw.start();
#pragma omp parallel num_threads(4)
        ParallelMatchBoyerMoore(text, N, pattern[i], M);
        sw.stop();
        eTimeBoyer = sw.elapsedTime();
        countBoyer = global_count;

        printf("|%-30s|%-10d|%-10.4lf|%-10d|%-10.4lf|\n", pattern[i], countNaive, eTimeNaive, countBoyer, eTimeBoyer);
    }
    printf("+------------------------------+---------------------+---------------------+\n");
    printf("\n\n");

}



void ParallelMatchNaive(const char* txt, int N, const char* pat, int M) {
    int thread_id = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    int local_count = 0;

    int size = N / thread_count;
    int end_index = (thread_id * size) + (size - 1);
    int start_index = 0;

    if (thread_id != 0) {
        start_index = (thread_id * size) - (M - 1);
        N = size + (M - 1);
    }
    else {
        start_index = thread_id * size;
        if (thread_id != thread_count - 1) {
            N = size;
        }
        if (thread_id == thread_count - 1) {
            end_index = N;
        }
    }

    for (int i = start_index; i <= end_index - M + 1; i++) {
        int j = 0;
        while ((j < M) && (txt[start_index + j] == pat[j])) {
            j = j + 1;
        }
        if (j == M) {
            local_count = local_count + 1;
        }
        start_index += 1;
    }

#  pragma omp critical
    global_count += local_count;
}

void ParallelMatchBoyerMoore(const char* txt, int N, const char* pat, int M) {
    int thread_id = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    int size = N / thread_count;
    int start_index = thread_id * size;
    int end_index = (thread_id * size) + (size - 1);
    int local_count = 0;

    if (thread_id == thread_count - 1) {
        end_index = N;
    }
    if (thread_id != 0) {
        start_index = (thread_id * size) - (M - 1);
    }
    else {
        start_index = thread_id * size;
        if (thread_id == thread_count - 1) {
            end_index = N;
        }
    }
    local_count = BoyerMooreMatchWithRange(txt, start_index, end_index, pat, M, badchar);

# pragma omp critical
    global_count += local_count;
}

char* loadFile(const char* fileName, int& fsize) {
    ifstream file;
    file.open(fileName);
    if (!file.is_open()) {
        cout << "File [" << fileName << "] could not open!" << endl;
        return nullptr;
    }
    const auto begin = file.tellg();
    file.seekg(0, ios::end);
    const auto end = file.tellg();
    fsize = (end - begin);
    char* content = new char[fsize + 1];
    file.seekg(0, ios::beg);
    file.read(content, fsize);
    file.close();
    content[fsize] = '\0';
    return content;
}
