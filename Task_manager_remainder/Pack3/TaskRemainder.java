package Pack3;

import Pack1.Task;
import Pack2.TaskManager;

import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.Scanner;

public class TaskRemainder {
    private static final String FILE_PATH = "C:/Users/krish/OneDrive/Desktop/Final-mid/Text_File_Handling/tasks.txt";

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        TaskManager taskManager = new TaskManager();
        int numberOfTasks = 0;

        if (args.length > 0) {
            try {
                numberOfTasks = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.out.println("Invalid command-line argument. Using interactive mode.");
            }
        }

        // Load the JDBC driver
        try {
            Class.forName("com.mysql.cj.jdbc.Driver"); // Replace with your database driver
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            return;
        }

        // Establish a connection to the database
        try (Connection connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/java", "root", "1234")) {
            taskManager.readTasksFromFile(FILE_PATH);

            for (int i = 0; i < numberOfTasks; i++) {
                processTaskMenu(scanner, taskManager, connection);
            }

            while (true) {
                processTaskMenu(scanner, taskManager, connection);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    private static void processTaskMenu(Scanner scanner, TaskManager taskManager, Connection connection) {
        System.out.println("Task Reminder Application");
        System.out.println("1. Create a task");
        System.out.println("2. View Task IDs");
        System.out.println("3. View tasks");
        System.out.println("4. Mark task as completed");
        System.out.println("5. Exit");

        System.out.print("Enter your choice: ");
        int choice = scanner.nextInt();
        scanner.nextLine();

        switch (choice) {
            case 1:
                createTask(scanner, taskManager, connection);
                break;

            case 2:
                if (taskManager.getTaskCount() == 0) {
                    System.out.println("No tasks to get Task Id. Please create a task first.");
                } else {
                    taskManager.viewAllTaskIds();
                }
                break;

           case 3:

            taskManager.viewTasksSubMenu(scanner, connection);
            break;

            case 4:
                if (taskManager.getTaskCount() == 0) {
                    System.out.println("No tasks to mark as completed. Please create a task first.");
                } else {
                    markTaskAsCompleted(scanner, taskManager, connection);
                }
                break;

            case 5:
                System.out.println("Exiting the application...");
                scanner.close();
                System.exit(0);
                break;

            default:
                System.out.println("Invalid choice! Please try again.");
                break;
        }
    }
    
    
    private static void createTask(Scanner scanner, TaskManager taskManager, Connection connection) {
        System.out.print("Enter task name: ");
        String name = scanner.nextLine();

        System.out.print("Enter due date (yyyy-mm-dd): ");
        LocalDate dueDate = LocalDate.parse(scanner.nextLine());
        System.out.print("Enter due time (hh:mm:ss): ");
        LocalTime dueTime = LocalTime.parse(scanner.nextLine());

        LocalDateTime dueDateTime = LocalDateTime.of(dueDate, dueTime);
        // Check if the due date and time are in the past
        if (dueDateTime.isBefore(LocalDateTime.now())) {
            System.out.println("Please re-enter Reminder. Reminder cannot be set in the past.");
            return; // Do not create the task
        }

        System.out.print("Enter category: ");
        String category = scanner.nextLine();
        System.out.print("Enter priority (1: Low, 2: Medium, 3: High): ");
        int priority = scanner.nextInt();
        scanner.nextLine();
        Task newTask = new Task(name, dueDate, dueTime, category, priority);
        taskManager.addTask(newTask);

        int taskId = taskManager.getTaskCount();
        System.out.println("Task created successfully with Task ID: " + taskId);


        taskManager.insertTaskIntoDatabase(newTask, connection, false); // Not completed by default
        

    System.out.print("Save the task to the file? (yes/no): ");
    String saveToFileChoice = scanner.nextLine().toLowerCase();        

       if ("yes".equals(saveToFileChoice)) {
        taskManager.writeTasksToFile(FILE_PATH);
        System.out.println("Tasks saved to file: " + FILE_PATH);
    }

    }

    private static void markTaskAsCompleted(Scanner scanner, TaskManager taskManager, Connection connection) {
        System.out.print("Enter the Task ID of the task to mark as completed: ");
        int taskId = scanner.nextInt();
        scanner.nextLine();

        if (taskId > 0 && taskId <= taskManager.getTaskCount()) {
            taskManager.markTaskAsCompleted(taskId - 1);

           
            taskManager.updateCompletedStatusInDatabase(taskId, true, connection);

            System.out.println("Task marked as completed!");
        } else {
            System.out.println("Invalid Task ID. Please enter a valid Task ID.");
        }
    }
    }
