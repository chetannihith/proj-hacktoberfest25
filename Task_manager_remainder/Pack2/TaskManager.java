package Pack2;

import Pack1.Task;
import java.util.Scanner;
import java.io.*;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.ResultSet;
import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.Arrays;
import java.time.LocalDateTime;

public class TaskManager {
    private Task[] tasks;
    private int size;

    public TaskManager() {
        this.tasks = new Task[10];
        this.size = 0;
    }

    public void addTask(Task task) {
        if (size == tasks.length) {
            tasks = Arrays.copyOf(tasks, size * 2);
        }
        tasks[size++] = task;
    }

    public void markTaskAsCompleted(int index) {
        if (index >= 0 && index < size) {
            tasks[index].setCompleted(true);
        }
    }

    public Task getTask(int index) {
        if (index >= 0 && index < size) {
            return tasks[index];
        }
        return null;
    }

    public int getTaskCount() {
        return size;
    }

    public Duration getRemainingTime(int index) {
        if (index >= 0 && index < size) {
            Task task = tasks[index];
            LocalDateTime dueDateTime = LocalDateTime.of(task.getDueDate(), task.getDueTime());
            return Duration.between(LocalDateTime.now(), dueDateTime);
        }
        return null;
    }

    public void viewAllTaskIds() {
        System.out.println("All Task IDs:");
        System.out.println("--------------------------------------------------");
        for (int j = 0; j < size; j++) {
            Task task = getTask(j);
            System.out.println("Task ID: " + (j + 1) + ", Task Name: " + task.getName());
        }
        System.out.println("--------------------------------------------------");
    }

    public void viewAllTasks(Connection connection) {
        System.out.println("All tasks:");
        System.out.println("--------------------------------------------------");
        for (int j = 0; j < size; j++) {
            Task task = getTask(j);
            printTaskDetails(j + 1, task);
        }
        System.out.println("--------------------------------------------------");
    }
    
   public void viewTasksSubMenu(Scanner scanner, Connection connection) {
        System.out.println("View Tasks Submenu");
        System.out.println("1. View all tasks");
        System.out.println("2. View task by task ID");
        System.out.print("Enter your choice: ");
        int subChoice = scanner.nextInt();
        scanner.nextLine();

        switch (subChoice) {
            case 1:
                viewAllTasks(connection);
                break;

            case 2:
                System.out.print("Enter Task ID to view: ");
                int taskId = scanner.nextInt();
                scanner.nextLine();
                viewTaskById(taskId);
                break;

            default:
                System.out.println("Invalid choice! Returning to main menu.");
                break;
        }
    }
    
    private void viewTaskById(int taskId) {
        if (taskId > 0 && taskId <= size) {
            Task task = getTask(taskId - 1);
            printTaskDetails(taskId, task);
        } else {
            System.out.println("Invalid Task ID. Please enter a valid Task ID.");
        }
    }
    private void printTaskDetails(int taskId, Task task) {
        System.out.println("Task ID: " + taskId);
        System.out.println("Task Name: " + task.getName());
        System.out.println("Due Date: " + task.getDueDate());
        System.out.println("Due Time: " + task.getDueTime());
        System.out.println("Category: " + task.getCategory());
        System.out.println("Priority: " + task.getPriority());
        System.out.println("Completed: " + (task.isCompleted() ? "Yes" : "No"));
        Duration remainingTime = getRemainingTime(taskId - 1);
        System.out.println("Remaining Time: " + remainingTime.toHours() + " hours "
                + remainingTime.toMinutes() % 60 + " minutes" + remainingTime.toSeconds() % 60 + " seconds");
    }

    public void insertTaskIntoDatabase(Task task, Connection connection, boolean completed) {
        
        String insertQuery = "INSERT INTO tasks (name, due_date, due_time, category, priority, completed) VALUES (?, ?, ?, ?, ?, ?)";
        try (PreparedStatement preparedStatement = connection.prepareStatement(insertQuery, PreparedStatement.RETURN_GENERATED_KEYS)) {
            preparedStatement.setString(1, task.getName());
            preparedStatement.setObject(2, task.getDueDate());
            preparedStatement.setObject(3, task.getDueTime());
            preparedStatement.setString(4, task.getCategory());
            preparedStatement.setInt(5, task.getPriority());
            preparedStatement.setBoolean(6, completed);

            int affectedRows = preparedStatement.executeUpdate();
            if (affectedRows > 0) {
                // Retrieve the auto-generated key (taskId)
                try (ResultSet generatedKeys = preparedStatement.getGeneratedKeys()) {
                    if (generatedKeys.next()) {
                        int taskId = generatedKeys.getInt(1);
                        // System.out.println("Task created successfully with Task ID: " + taskId);
                    } else {
                        System.out.println("Failed to retrieve taskId");
                    }
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public void updateCompletedStatusInDatabase(int taskId, boolean completed, Connection connection) {
      
        String updateQuery = "UPDATE tasks SET completed = ? WHERE taskId = ?";
        try (PreparedStatement preparedStatement = connection.prepareStatement(updateQuery)) {
            preparedStatement.setBoolean(1, completed);
            preparedStatement.setInt(2, taskId);
            preparedStatement.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
  
    public void loadTasksFromDatabase(Connection connection) {
        tasks = new Task[10];
        size = 0;


        String sql = "SELECT * FROM tasks";

        try (PreparedStatement preparedStatement = connection.prepareStatement(sql)) {
            try (ResultSet resultSet = preparedStatement.executeQuery()) {
                while (resultSet.next()) {
                    String name = resultSet.getString("name");
                    LocalDate dueDate = resultSet.getDate("due_date").toLocalDate();
                    LocalTime dueTime = resultSet.getTime("due_time").toLocalTime();
                    String category = resultSet.getString("category");
                    int priority = resultSet.getInt("priority");
                    boolean isCompleted = resultSet.getBoolean("completed");

                    Task task = new Task(name, dueDate, dueTime, category, priority);
                    task.setCompleted(isCompleted);
                    addTask(task);
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public void writeTasksToFile(String filePath) {
        try (ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream(filePath))) {
            for (int i = 0; i < size; i++) {
                outputStream.writeObject(tasks[i]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void readTasksFromFile(String filePath) {
        try (ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(filePath))) {
            Task task;
            while ((task = (Task) inputStream.readObject()) != null) {
                addTask(task);
            }
        } catch (EOFException e) {
     
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
}
