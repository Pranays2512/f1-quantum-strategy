package com.taskscheduler.taskservice.controller;

import com.taskscheduler.taskservice.dto.TaskDTO;
import com.taskscheduler.taskservice.dto.UserDTO;
import com.taskscheduler.taskservice.service.TaskService;
import com.taskscheduler.taskservice.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/tasks")
public class TaskController {

    @Autowired
    private TaskService taskService;

    @Autowired
    private UserService userService;

    private UserDTO getUserFromToken(String authHeader) throws Exception {
        return userService.getUserFromToken(authHeader);
    }

    @PostMapping("/add")
    public ResponseEntity<?> addTask(@RequestBody TaskDTO task, @RequestHeader("Authorization") String authHeader) {
        try {
            UserDTO user = getUserFromToken(authHeader);
            TaskDTO savedTask = taskService.addTask(task, user.getId());
            return new ResponseEntity<>(savedTask, HttpStatus.CREATED);
        } catch (RuntimeException e) {
            return new ResponseEntity<>("Authentication error: " + e.getMessage(), HttpStatus.UNAUTHORIZED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error adding task: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/all")
    public ResponseEntity<?> getAllTasks(@RequestHeader("Authorization") String authHeader) {
        try {
            UserDTO user = getUserFromToken(authHeader);
            List<TaskDTO> tasks = taskService.getAllTasksByUser(user.getId());
            return new ResponseEntity<>(tasks, HttpStatus.OK);
        } catch (RuntimeException e) {
            return new ResponseEntity<>("Authentication error: " + e.getMessage(), HttpStatus.UNAUTHORIZED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error fetching tasks: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/get/{id}")
    public ResponseEntity<?> getTaskById(@PathVariable Long id, @RequestHeader("Authorization") String authHeader) {
        try {
            UserDTO user = getUserFromToken(authHeader);
            Optional<TaskDTO> taskOptional = taskService.getTaskByIdAndUser(id, user.getId());
            if (taskOptional.isPresent()) {
                return new ResponseEntity<>(taskOptional.get(), HttpStatus.OK);
            } else {
                return new ResponseEntity<>("Task not found or access denied", HttpStatus.NOT_FOUND);
            }
        } catch (RuntimeException e) {
            return new ResponseEntity<>("Authentication error: " + e.getMessage(), HttpStatus.UNAUTHORIZED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error fetching task: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PutMapping("/edit/{id}")
    public ResponseEntity<?> updateTask(@PathVariable Long id, @RequestBody TaskDTO task, @RequestHeader("Authorization") String authHeader) {
        try {
            UserDTO user = getUserFromToken(authHeader);
            Optional<TaskDTO> updatedTaskOptional = taskService.updateTask(id, task, user.getId());
            if (updatedTaskOptional.isPresent()) {
                return new ResponseEntity<>(updatedTaskOptional.get(), HttpStatus.OK);
            } else {
                return new ResponseEntity<>("Task not found or access denied", HttpStatus.NOT_FOUND);
            }
        } catch (RuntimeException e) {
            return new ResponseEntity<>("Authentication error: " + e.getMessage(), HttpStatus.UNAUTHORIZED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error updating task: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @DeleteMapping("/delete/{id}")
    public ResponseEntity<String> deleteTask(@PathVariable Long id, @RequestHeader("Authorization") String authHeader) {
        try {
            UserDTO user = getUserFromToken(authHeader);
            boolean deleted = taskService.deleteTask(id, user.getId());
            if (deleted) {
                return new ResponseEntity<>("Task deleted successfully", HttpStatus.OK);
            } else {
                return new ResponseEntity<>("Task not found or access denied", HttpStatus.NOT_FOUND);
            }
        } catch (RuntimeException e) {
            return new ResponseEntity<>("Authentication error: " + e.getMessage(), HttpStatus.UNAUTHORIZED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error deleting task: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PutMapping("/toggle/{id}")
    public ResponseEntity<?> toggleTaskStatus(@PathVariable Long id, @RequestHeader("Authorization") String authHeader) {
        try {
            UserDTO user = getUserFromToken(authHeader);
            Optional<TaskDTO> updatedTaskOptional = taskService.toggleTaskStatus(id, user.getId());

            if (updatedTaskOptional.isPresent()) {
                return new ResponseEntity<>(updatedTaskOptional.get(), HttpStatus.OK);
            } else {
                return new ResponseEntity<>("Task not found or access denied", HttpStatus.NOT_FOUND);
            }
        } catch (RuntimeException e) {
            return new ResponseEntity<>("Authentication error: " + e.getMessage(), HttpStatus.UNAUTHORIZED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error updating task status: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/status/{done}")
    public ResponseEntity<?> getTasksByStatus(@PathVariable boolean done, @RequestHeader("Authorization") String authHeader) {
        try {
            UserDTO user = getUserFromToken(authHeader);
            List<TaskDTO> tasks = taskService.getTasksByStatus(done, user.getId());
            return new ResponseEntity<>(tasks, HttpStatus.OK);
        } catch (RuntimeException e) {
            return new ResponseEntity<>("Authentication error: " + e.getMessage(), HttpStatus.UNAUTHORIZED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error fetching tasks by status: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/priority/{priority}")
    public ResponseEntity<?> getTasksByPriority(@PathVariable String priority, @RequestHeader("Authorization") String authHeader) {
        try {
            UserDTO user = getUserFromToken(authHeader);
            List<TaskDTO> tasks = taskService.getTasksByPriority(priority, user.getId());
            return new ResponseEntity<>(tasks, HttpStatus.OK);
        } catch (RuntimeException e) {
            return new ResponseEntity<>("Authentication error: " + e.getMessage(), HttpStatus.UNAUTHORIZED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error fetching tasks by priority: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/category/{category}")
    public ResponseEntity<?> getTasksByCategory(@PathVariable String category, @RequestHeader("Authorization") String authHeader) {
        try {
            UserDTO user = getUserFromToken(authHeader);
            List<TaskDTO> tasks = taskService.getTasksByCategory(category, user.getId());
            return new ResponseEntity<>(tasks, HttpStatus.OK);
        } catch (RuntimeException e) {
            return new ResponseEntity<>("Authentication error: " + e.getMessage(), HttpStatus.UNAUTHORIZED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error fetching tasks by category: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/overdue")
    public ResponseEntity<?> getOverdueTasks(@RequestHeader("Authorization") String authHeader) {
        try {
            UserDTO user = getUserFromToken(authHeader);
            List<TaskDTO> tasks = taskService.getOverdueTasks(user.getId());
            return new ResponseEntity<>(tasks, HttpStatus.OK);
        } catch (RuntimeException e) {
            return new ResponseEntity<>("Authentication error: " + e.getMessage(), HttpStatus.UNAUTHORIZED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error fetching overdue tasks: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/today")
    public ResponseEntity<?> getTodayTasks(@RequestHeader("Authorization") String authHeader) {
        try {
            UserDTO user = getUserFromToken(authHeader);
            List<TaskDTO> tasks = taskService.getTodayTasks(user.getId());
            return new ResponseEntity<>(tasks, HttpStatus.OK);
        } catch (RuntimeException e) {
            return new ResponseEntity<>("Authentication error: " + e.getMessage(), HttpStatus.UNAUTHORIZED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error fetching today's tasks: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/starting-soon")
    public ResponseEntity<?> getTasksStartingSoon(@RequestHeader("Authorization") String authHeader) {
        try {
            UserDTO user = getUserFromToken(authHeader);
            List<TaskDTO> tasks = taskService.getTasksStartingSoon(user.getId());
            return new ResponseEntity<>(tasks, HttpStatus.OK);
        } catch (RuntimeException e) {
            return new ResponseEntity<>("Authentication error: " + e.getMessage(), HttpStatus.UNAUTHORIZED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error fetching upcoming tasks: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/statistics")
    public ResponseEntity<?> getTaskStatistics(@RequestHeader("Authorization") String authHeader) {
        try {
            UserDTO user = getUserFromToken(authHeader);
            TaskService.TaskStatistics stats = taskService.getTaskStatistics(user.getId());
            return new ResponseEntity<>(stats, HttpStatus.OK);
        } catch (RuntimeException e) {
            return new ResponseEntity<>("Authentication error: " + e.getMessage(), HttpStatus.UNAUTHORIZED);
        } catch (Exception e) {
            return new ResponseEntity<>("Error fetching statistics: " + e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }
}
