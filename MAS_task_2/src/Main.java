import entities.Entity;
import grid.Cell;
import grid.Step;

int K = 6;

Game game = new Game(K);

void main(){


    game.setCellEntity(Entity.CLIFF, 0, 4);
    game.setCellEntity(Entity.CLIFF, 2, 1);
    game.setCellEntity(Entity.CLIFF,2,  5);
    game.setCellEntity(Entity.CLIFF, 3, 3);

    game.setCellEntity(Entity.RESOURCE_1,0, 3 );
    game.setCellEntity(Entity.RESOURCE_1,0, 5 );
    game.setCellEntity(Entity.RESOURCE_1,3, 1 );
    game.setCellEntity(Entity.RESOURCE_1,3, 5 );
    game.setCellEntity(Entity.RESOURCE_1,4, 3 );



    do {
        drawGrid();

        drawInfo();

        drawOptions();

        handlePlayerInput();
    }
    while (game.isActive());

    printPath();
}

private void drawGrid() {
    System.out.println("\n=== GAME GRID ===");

    Cell[][] grid = game.getGrid();
    int playerX = game.getPlayerX();
    int playerY = game.getPlayerY();
    int K = grid.length;

    // Print column headers
    System.out.print("\t");
    for (int j = 0; j < K; j++) {
        System.out.print(j + 1 + "\t");
    }
    System.out.println();

    // Print rows in reverse (bottom-left is start)
    for (int i = K - 1; i >= 0; i--) {
        System.out.print(i + 1 + "\t");

        for (int j = 0; j < K; j++) {
            Cell cell = grid[i][j];
            String symbol = getCellSymbol(cell, i, j, playerX, playerY, grid, K);
            System.out.print(symbol + "\t");
        }
        System.out.println();
    }
}

private String getCellSymbol(Cell cell, int row, int col, int playerX, int playerY, Cell[][] grid, int K) {
    // Player position always visible

    if (row == playerY && col == playerX) {
        if (game.isAdjacentToCliff(row, col)) {
            return "P!";
        }
        return "P";
    }

    // Check if current cell is visited
    if (cell.isVisited()) {
        Entity entity = cell.getEntity();
        if (entity == Entity.CLIFF) {
            return "C";
        } else if (entity == Entity.RESOURCE_1) {
            return "R1";
        } else if (game.isAdjacentToCliff(row, col)) {
            return "!";
        }
        return "*";
    }

    // Cell not visited - check if adjacent cell is visited
//    if (isAdjacentVisited(row, col, grid, K)) {
//        Entity entity = cell.getEntity();
//
//        // Only reveal CLIFF (C) if adjacent cell is visited
//        if (entity == Entity.CLIFF) {
//            return "C";
//        } else if (entity == Entity.RESOURCE_1) {
//            // R1 remains hidden
//            return "?";
//        }
//        return ".";
//    }

    // Not visited and no adjacent visited cells - completely hidden
    return "?";
}




private boolean isAdjacentVisited(int row, int col, Cell[][] grid, int K) {
    // Check all 4 adjacent cells (up, down, left, right)
    int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    for (int[] dir : directions) {
        int newRow = row + dir[0];
        int newCol = col + dir[1];

        // Check if within bounds
        if (newRow >= 0 && newRow < K && newCol >= 0 && newCol < K) {
            if (grid[newRow][newCol].isVisited()) {
                return true;
            }
        }
    }
    return false;
}

private void drawInfo() {
    System.out.println("\n=== GAME INFO ===");
    System.out.println("Time: " + game.getCurrentTime() + "/50");
    System.out.println("Score: " + game.getScore());
    System.out.println("Player Position: (" + (game.getPlayerX() + 1) + ", " + (game.getPlayerY() + 1) + ")");
    System.out.println("Game Active: " + game.isActive());
}

private void drawOptions() {

    System.out.println("W - Move Up (Y+1)");
    System.out.println("S - Move Down (Y-1)");
    System.out.println("A - Move Left (X-1)");
    System.out.println("D - Move Right (X+1)");
}

private void printPath(){
    for(Step step: game.getPath()){
        System.out.println(step);
    }
}

private void handlePlayerInput() {
    Scanner scanner = new Scanner(System.in);
    System.out.print("\nEnter command: ");
    String input = scanner.nextLine().toUpperCase();

    switch (input) {
        case "W":
            game.movePlayerY(true);
            break;
        case "S":
            game.movePlayerY(false);
            break;
        case "A":
            game.movePlayerX(false);
            break;
        case "D":
            game.movePlayerX(true);
            break;
        case "B":

        case "Q":
            game.setActive(false);
            System.out.println("Game ended!");
            break;
        default:
            System.out.println("Invalid command!");
    }
}
