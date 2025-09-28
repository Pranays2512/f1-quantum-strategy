package com.taskscheduler.taskservice.service;

import com.taskscheduler.taskservice.dto.TaskDTO;
import com.taskscheduler.taskservice.entity.Task;
import com.taskscheduler.taskservice.repository.TaskRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class TaskService {

    @Autowired
    private TaskRepository taskRepository;

    private TaskDTO convertToDTO(Task task) {
        return new TaskDTO(
                task.getId(),
                task.getDescription(),
                task.getStartTime(),
                task.getEndTime(),
                task.getDone(),
                task.getPriority(),
                task.getCategory(),
                task.getNotes()
        );
    }

    private Task convertToEntity(TaskDTO taskDTO, Long userId) {
        Task task = new Task();
        task.setId(taskDTO.getId());
        task.setDescription(taskDTO.getDescription());
        task.setStartTime(taskDTO.getStartTime());
        task.setEndTime(taskDTO.getEndTime());
        task.setDone(taskDTO.getDone() != null ? taskDTO.getDone() : false);
        task.setPriority(taskDTO.getPriority() != null ? taskDTO.getPriority() : "Medium");
        task.setCategory(taskDTO.getCategory() != null ? taskDTO.getCategory() : "Personal");
        task.setNotes(taskDTO.getNotes());
        task.setUserId(userId);
        return task;
    }

    private List<TaskDTO> convertToDTOList(List<Task> tasks) {
        return tasks.stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    public TaskDTO addTask(TaskDTO taskDTO, Long userId) throws Exception {
        Task task = convertToEntity(taskDTO, userId);
        task.setDone(false);
        task.setId(null);
        Task savedTask = taskRepository.save(task);
        return convertToDTO(savedTask);
    }

    public List<TaskDTO> getAllTasksByUser(Long userId) throws Exception {
        List<Task> tasks = taskRepository.findByUserIdOrderByPriorityAscEndTimeAsc(userId);

        tasks.sort(
                Comparator
                        .comparingInt((Task t) -> {
                            switch (t.getPriority()) {
                                case "High": return 1;
                                case "Medium": return 2;
                                case "Low": return 3;
                                default: return 4;
                            }
                        })
                        .thenComparing(Task::getEndTime)
        );

        return convertToDTOList(tasks);
    }

    public Optional<TaskDTO> getTaskByIdAndUser(Long id, Long userId) throws Exception {
        return taskRepository.findByIdAndUserId(id, userId).map(this::convertToDTO);
    }

    public Optional<TaskDTO> updateTask(Long id, TaskDTO taskDetails, Long userId) throws Exception {
        return taskRepository.findByIdAndUserId(id, userId).map(existingTask -> {
            existingTask.setDescription(taskDetails.getDescription());
            existingTask.setStartTime(taskDetails.getStartTime());
            existingTask.setEndTime(taskDetails.getEndTime());
            existingTask.setPriority(taskDetails.getPriority());
            existingTask.setCategory(taskDetails.getCategory());
            existingTask.setNotes(taskDetails.getNotes());

            if (taskDetails.getDone() != null) {
                existingTask.setDone(taskDetails.getDone());
            }
            return convertToDTO(taskRepository.save(existingTask));
        });
    }

    public boolean deleteTask(Long id, Long userId) throws Exception {
        if (taskRepository.existsByIdAndUserId(id, userId)) {
            taskRepository.deleteById(id);
            return true;
        }
        return false;
    }

    public Optional<TaskDTO> toggleTaskStatus(Long id, Long userId) throws Exception {
        return taskRepository.findByIdAndUserId(id, userId).map(task -> {
            task.setDone(!task.getDone());
            return convertToDTO(taskRepository.save(task));
        });
    }

    public List<TaskDTO> getTasksByStatus(boolean done, Long userId) throws Exception {
        return convertToDTOList(taskRepository.findByDoneAndUserId(done, userId));
    }

    public List<TaskDTO> getTasksByPriority(String priority, Long userId) throws Exception {
        return convertToDTOList(taskRepository.findByPriorityAndUserId(priority, userId));
    }

    public List<TaskDTO> getTasksByCategory(String category, Long userId) throws Exception {
        return convertToDTOList(taskRepository.findByCategoryAndUserId(category, userId));
    }

    public List<TaskDTO> getOverdueTasks(Long userId) throws Exception {
        return convertToDTOList(taskRepository.findOverdueTasksByUserId(userId, LocalDateTime.now()));
    }

    public List<TaskDTO> getTodayTasks(Long userId) throws Exception {
        LocalDateTime startOfDay = LocalDate.now().atStartOfDay();
        LocalDateTime endOfDay = LocalDate.now().atTime(LocalTime.MAX);
        return convertToDTOList(taskRepository.findTasksBetweenByUserId(userId, startOfDay, endOfDay));
    }

    public List<TaskDTO> getTasksStartingSoon(Long userId) throws Exception {
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime oneHourLater = now.plusHours(1);
        return convertToDTOList(taskRepository.findTasksStartingSoonByUserId(userId, now, oneHourLater));
    }

    public TaskStatistics getTaskStatistics(Long userId) throws Exception {
        long totalTasks = taskRepository.countByUserIdAndDone(userId, true) + taskRepository.countByUserIdAndDone(userId, false);
        long completedTasks = taskRepository.countByUserIdAndDone(userId, true);
        long pendingTasks = taskRepository.countByUserIdAndDone(userId, false);
        long highPriorityTasks = taskRepository.countByUserIdAndPriority(userId, "High");
        long workTasks = taskRepository.countByUserIdAndCategory(userId, "Work");
        long personalTasks = taskRepository.countByUserIdAndCategory(userId, "Personal");

        LocalDateTime startOfDay = LocalDate.now().atStartOfDay();
        LocalDateTime endOfDay = LocalDate.now().atTime(LocalTime.MAX);
        long todayTasks = taskRepository.countTasksBetweenByUserId(userId, startOfDay, endOfDay);
        long overdueTasks = taskRepository.countOverdueTasksByUserId(userId, LocalDateTime.now());

        return new TaskStatistics(totalTasks, completedTasks, pendingTasks,
                highPriorityTasks, workTasks, personalTasks,
                todayTasks, overdueTasks);
    }

    public static class TaskStatistics {
        private final long totalTasks;
        private final long completedTasks;
        private final long pendingTasks;
        private final long highPriorityTasks;
        private final long workTasks;
        private final long personalTasks;
        private final long todayTasks;
        private final long overdueTasks;

        public TaskStatistics(long totalTasks, long completedTasks, long pendingTasks,
                              long highPriorityTasks, long workTasks, long personalTasks,
                              long todayTasks, long overdueTasks) {
            this.totalTasks = totalTasks;
            this.completedTasks = completedTasks;
            this.pendingTasks = pendingTasks;
            this.highPriorityTasks = highPriorityTasks;
            this.workTasks = workTasks;
            this.personalTasks = personalTasks;
            this.todayTasks = todayTasks;
            this.overdueTasks = overdueTasks;
        }

        // Getters
        public long getTotalTasks() { return totalTasks; }
        public long getCompletedTasks() { return completedTasks; }
        public long getPendingTasks() { return pendingTasks; }
        public long getHighPriorityTasks() { return highPriorityTasks; }
        public long getWorkTasks() { return workTasks; }
        public long getPersonalTasks() { return personalTasks; }
        public long getTodayTasks() { return todayTasks; }
        public long getOverdueTasks() { return overdueTasks; }
    }
}