package Pack1;

import java.io.Serializable;
import java.time.LocalDate;
import java.time.LocalTime;

public class Task implements Serializable {
    private String name;
    private LocalDate dueDate;
    private LocalTime dueTime;
    private String category;
    private int priority;
    private boolean isCompleted;

    // Constructor
    public Task(String name, LocalDate dueDate, LocalTime dueTime, String category, int priority) {
        this.name = name;
        this.dueDate = dueDate;
        this.dueTime = dueTime;
        this.category = category;
        this.priority = priority;
        this.isCompleted = false;
    }

    // Getters and setters
    public boolean isCompleted() {
        return isCompleted;
    }

    public void setCompleted(boolean completed) {
        isCompleted = completed;
    }

    public String getName() {
        return name;
    }

    public LocalDate getDueDate() {
        return dueDate;
    }

    public LocalTime getDueTime() {
        return dueTime;
    }

    public String getCategory() {
        return category;
    }

    public int getPriority() {
        return priority;
    }

    
    public String toFileString() {
        return name + "|" + dueDate + "|" + dueTime + "|" + category + "|" + priority + "|" + (isCompleted ? "1" : "0");
    }

    
    public static Task fromFileString(String fileString) {
        String[] parts = fileString.split("\\|");
        if (parts.length == 6) {
            String name = parts[0];
            LocalDate dueDate = LocalDate.parse(parts[1]);
            LocalTime dueTime = LocalTime.parse(parts[2]);
            String category = parts[3];
            int priority = Integer.parseInt(parts[4]);
            boolean isCompleted = parts[5].equals("1");
            return new Task(name, dueDate, dueTime, category, priority);
        }
        return null;
    }
}
