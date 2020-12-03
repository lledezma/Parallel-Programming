#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

int NaiveMatch(const char* text, int N, const char* pattern, int M)
{
	int count = 0;
	if (N >= M) {
		for (int i = 0; i <= N - M; i++) {
			int j = 0;
			while ((j < M) && (text[i + j] == pattern[j])) {
				j = j + 1;
			}
			if (j == M) {
				count = count + 1;
			}
		}
	}
	return count;
}

int NaiveMatchWithRange(const char* text, int sIndex, int eIndex, const char* pattern, int M)
{
	/* IMPLEMENT THE FUNCTION WITH THE GIVEN ALGORITHM */
	int N = eIndex - sIndex;
	int fountCount = 0;
	for (int i = 0; i < N - M; ++i)
	{
		int j = 0;
		while ((j < M) && (text[sIndex + i + j] == pattern[j]))
		{
			j = j + 1;
		}
		if (j == M)
		{
			fountCount = fountCount + 1;
		}
	}
	return  fountCount;
}

# define NO_OF_CHARS 256  

// The preprocessing function for Boyer Moore's  
// bad character heuristic  
void preprocessHeuristicArray(const char* pattern, int M, int badchar[NO_OF_CHARS])
{
	/* IMPLEMENT THE FUNCTION WITH THE GIVEN ALGORITHM */
	int i;
	for (i = 0; i < NO_OF_CHARS; i++) {
		badchar[i] = -1;
	}
	for (i = 0; i < M; i++) {
		badchar[(int)pattern[i]] = i;
	}
}

/* A pattern searching function that uses Bad
Character Heuristic of Boyer Moore Algorithm */
#define max(x,y) ((x) >= (y)) ? (x) : (y)
int BoyerMooreMatch(const char* text, int N, const char* pattern, int M, int* badchar)
{
	/* IMPLEMENT THE FUNCTION WITH THE GIVEN ALGORITHM */
	int fountCount = 0;
	int s = 0;
	while (s <= (N - M))
	{
		int j = M - 1;
		while (j >= 0 && pattern[j] == text[s + j])
		{
			--j;
		}
		if (j < 0)
		{
			++fountCount;
			s += (s + M < N) ? M - badchar[text[s + M]] : 1;
		}
		else
		{
			s += max(1, j - badchar[text[s + j]]);
		}
	}
	return  fountCount;
}


int BoyerMooreMatchWithRange(const char* text, int sIndex, int eIndex, const char* pattern, int M, int* badchar)
{
	/* IMPLEMENT THE FUNCTION WITH THE GIVEN ALGORITHM */
	int N = eIndex - sIndex;
	int fountCount = 0;
	int s = 0;
	while (s <= (N - M))
	{
		int j = M - 1;
		while (j >= 0 && pattern[j] == text[sIndex + s + j])
		{
			--j;
		}
		if (j < 0)
		{
			++fountCount;
			s += (s + M < N) ? M - badchar[text[sIndex + s + M]] : 1;
		}
		else
		{
			s += max(1, j - badchar[text[sIndex + s + j]]);
		}
	}
	return  fountCount;
}