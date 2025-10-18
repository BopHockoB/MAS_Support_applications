package grid;

import entities.Entity;

import java.util.ArrayList;

public class Cell {
    public Cell(){
        entity = Entity.PLAIN;
    };

    public Cell(Entity e){
        entity = e;
    }

    private Entity entity;
    private boolean visited = false;

    public boolean isVisited() {
        return visited;
    }

    public void visitCell(){
        visited = true;
    }

    public void setVisited(boolean visited) {
        this.visited = visited;
    }

    public Entity getEntity() {
        return entity;
    }

    public void setEntity(Entity entity) {
        this.entity = entity;
    }



}
