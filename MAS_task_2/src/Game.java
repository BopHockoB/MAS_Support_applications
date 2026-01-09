import entities.Entity;
import grid.Cell;
import grid.Step;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;


import static entities.Entity.CLIFF;
import static entities.Entity.RESOURCE_1;


public class Game implements Cloneable{

    public Game(int K) {
       grid = new Cell[K][K];

        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                grid[i][j] = new Cell();
            }
        }

        grid[0][0].visitCell();
    }


    private Cell[][] grid;
    private int currentTime = 0;
    private int score = 0;
    private boolean isActive = true;

    private int playerX = 0;
    private int playerY = 0;

    private List<Step> path = new ArrayList<>();

    public void increaseTime(){
        currentTime += 1;
        if(currentTime == 50)
            isActive = false;
    }

    public void addScore(){
        this.score += 50 - currentTime;
        System.out.println("You found resource!!!");
    }

    public void movePlayerX(boolean direction){
        if(!isActive)
            return;
        int oldY = playerY;
        int oldX = playerX;


        if(direction && playerX < grid.length - 1)
            playerX += 1;
        else if (!direction && playerX > 0)
            playerX -= 1;


        changeGameStateAfterMove(oldX, oldY);
    }

    public void movePlayerY(boolean direction){
        if(!isActive)
            return;
        int oldY = playerY;
        int oldX = playerX;


        if(direction && playerY < grid.length - 1)
            playerY += 1;
        else if (!direction && playerY > 0)
            playerY -= 1;

        changeGameStateAfterMove(oldX, oldY);
    }

    private void changeGameStateAfterMove(int oldX, int oldY){
        increaseTime();

        var currentCell = grid[playerY][playerX];  // ✓ Changed order



        if(currentCell.getEntity() == CLIFF)
            isActive = false;
        else if (currentCell.getEntity() == RESOURCE_1 && !currentCell.isVisited())
            addScore();

        currentCell.visitCell();

        var step = new Step(
                currentTime,
                playerX, playerY,
                oldX, oldY,
                isAdjacentToCliff(playerY, playerX));
        path.add(step);
    }


    public void setCellEntity(Entity entity, int x, int y){
        grid[y][x].setEntity(entity);
    }

        public boolean isAdjacentToCliff(int row, int col) {
        int K = grid.length;

        // Check all 4 adjacent cells (up, down, left, right)
        int[][] directions = {
                {row + 1, col},     // Up
                {row - 1, col},     // Down
                {row, col + 1},     // Right
                {row, col - 1}      // Left
        };

        for (int[] dir : directions) {
            int r = dir[0];
            int c = dir[1];

            // Check if within bounds
            if (r >= 0 && r < K && c >= 0 && c < K) {
                if (grid[r][c].getEntity() == Entity.CLIFF) {
                    return true;
                }
            }
        }

        return false;
    }




    // Getters/Setters

    public Cell[][] getGrid() {
        return grid;
    }

    public int getCurrentTime() {
        return currentTime;
    }

    public void setCurrentTime(int currentTime) {
        this.currentTime = currentTime;
    }

    public boolean isActive() {
        return isActive;
    }

    public void setActive(boolean active) {
        isActive = active;
    }

    public int getScore() {
        return score;
    }

    public void setScore(int score) {
        this.score = score;
    }

    public int getPlayerY() {
        return playerY;
    }

    public void setPlayerY(int playerY) {
        this.playerY = playerY;
    }

    public int getPlayerX() {
        return playerX;
    }

    public void setPlayerX(int playerX) {
        this.playerX = playerX;
    }

    public List<Step> getPath() {
        return path;
    }

    public void setPath(List<Step> path) {
        this.path = path;
    }

    @Override
    public Object clone() throws CloneNotSupportedException {
        return super.clone();
    }
}
