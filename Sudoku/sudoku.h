// sudoku.h

#ifndef SUDOKU_H
#define SUDOKU_H

#define N 9

int checkForSafe(int grid[N][N], int row, int col, int num);
void print(int arr[N][N]);
int solveSudoku(int grid[N][N], int row, int col);

#endif
