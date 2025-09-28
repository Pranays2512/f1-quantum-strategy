package com.taskscheduler.taskservice.dto;

import java.time.LocalDateTime;

public class TaskDTO {
    private Long id;
    private String description;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private Boolean done;
    private String priority;
    private String category;
    private String notes;

    public TaskDTO() {}

    public TaskDTO(Long id, String description, LocalDateTime startTime,
                   LocalDateTime endTime, Boolean done, String priority,
                   String category, String notes) {
        this.id = id;
        this.description = description;
        this.startTime = startTime;
        this.endTime = endTime;
        this.done = done;
        this.priority = priority;
        this.category = category;
        this.notes = notes;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public LocalDateTime getStartTime() { return startTime; }
    public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }

    public LocalDateTime getEndTime() { return endTime; }
    public void setEndTime(LocalDateTime endTime) { this.endTime = endTime; }

    public Boolean getDone() { return done; }
    public void setDone(Boolean done) { this.done = done; }

    public String getPriority() { return priority; }
    public void setPriority(String priority) { this.priority = priority; }

    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }

    public String getNotes() { return notes; }
    public void setNotes(String notes) { this.notes = notes; }
}
