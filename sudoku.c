// sudoku.c

#include <stdio.h>
#include "sudoku.h"

int checkForSafe(int grid[N][N], int row, int col, int num) {
    for (int x = 0; x < N; x++)
        if (grid[row][x] == num || grid[x][col] == num)
            return 0;

    int startRow = row - row % 3, startCol = col - col % 3;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (grid[i + startRow][j + startCol] == num)
                return 0;

    return 1;
}

void print(int arr[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d", arr[i][j]);
        printf("\n");
    }
}

int solveSudoku(int grid[N][N], int row, int col) {
    if (row == N)
        return 1;  // Entire grid solved

    if (col == N) {
        row++;
        col = 0;
    }

    if (grid[row][col] > 0)
        return solveSudoku(grid, row, col + 1);

    for (int num = 1; num <= N; num++) {
        if (checkForSafe(grid, row, col, num) == 1) {
            grid[row][col] = num;

            if (solveSudoku(grid, row, col + 1) == 1)
                return 1;

            grid[row][col] = 0; // Backtrack if the solution is not found
        }
    }

    return 0;
}
