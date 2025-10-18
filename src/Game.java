import entities.Entity;
import grid.Cell;

import java.util.Set;

import static entities.Entity.CLIFF;
import static entities.Entity.RESOURCE_1;


public class Game {

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


    public void increaseTime(){
        currentTime += 1;
        if(currentTime == 50)
            isActive = false;
    }

    public void addScore(){
        this.score += 50 - currentTime;
    }

    public void movePlayerX(boolean direction){
        if(!isActive)
            return;

        if(direction && playerX < grid.length - 1)
            playerX += 1;
        else if (!direction && playerX > 0)
            playerX -= 1;

        changeGameStateAfterMove();

    }

    public void movePlayerY(boolean direction){
        if(!isActive)
            return;

        if(direction && playerY < grid.length - 1)
            playerY += 1;
        else if (!direction && playerY > 0)
            playerY -= 1;


        changeGameStateAfterMove();
    }

    private void changeGameStateAfterMove(){
        increaseTime();

        var currentCell = grid[playerY][playerX];  // ✓ Changed order

        currentCell.visitCell();

        if(currentCell.getEntity() == CLIFF)
            isActive = false;
        else if (currentCell.getEntity() == RESOURCE_1)
            addScore();
    }


    public void setCellEntity(Entity entity, int x, int y){
        grid[y][x].setEntity(entity);
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
}
