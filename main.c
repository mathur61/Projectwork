// main.c

#include <stdio.h>
#include "sudoku.h"

#define MAX_FILE_PATH_LENGTH 100

int main(int argc, char *argv[]) {
    if (argc > 2) {
        printf("Usage: %s [optional_file_path]\n", argv[0]);
        return 1;
    }

    char file_path[MAX_FILE_PATH_LENGTH];
    if (argc == 2) {
        // Use the provided file path
        snprintf(file_path, sizeof(file_path), "%s", argv[1]);
    } else {
        // Default file path if not provided as an argument
        snprintf(file_path, sizeof(file_path), "inputs/testx");
    }

    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        printf("Error: File '%s' not found.\n", file_path);
        return 1;
    }

    int grid[N][N];

    // Read the puzzle from the file
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            char cell;
            if (fscanf(file, " %c", &cell) != 1) {
                printf("Error reading puzzle from file.\n");
                fclose(file);
                return 1;
            }
            grid[i][j] = (cell == '-') ? -1 : (cell - '0');
        }
    }

    fclose(file);

    if (solveSudoku(grid, 0, 0)) {
        print(grid);
    } else {
        printf("Cannot backtrack! Solution does not exist!\n");
    }

    return 0;
}
