package com.taskscheduler.taskservice.repository;

import com.taskscheduler.taskservice.entity.Task;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface TaskRepository extends JpaRepository<Task, Long> {

    // Find all tasks for a specific user
    List<Task> findByUserIdOrderByPriorityAscEndTimeAsc(Long userId);

    // Find task by ID and user ID
    Optional<Task> findByIdAndUserId(Long id, Long userId);

    // Find tasks by status and user
    List<Task> findByDoneAndUserId(Boolean done, Long userId);

    // Find tasks by priority and user
    List<Task> findByPriorityAndUserId(String priority, Long userId);

    // Find tasks by category and user
    List<Task> findByCategoryAndUserId(String category, Long userId);

    // Custom queries for user-specific data
    @Query("SELECT t FROM Task t WHERE t.userId = :userId AND t.done = false AND t.endTime < :currentTime")
    List<Task> findOverdueTasksByUserId(@Param("userId") Long userId, @Param("currentTime") LocalDateTime currentTime);

    @Query("SELECT t FROM Task t WHERE t.userId = :userId AND t.done = false AND t.startTime BETWEEN :now AND :future")
    List<Task> findTasksStartingSoonByUserId(@Param("userId") Long userId, @Param("now") LocalDateTime now, @Param("future") LocalDateTime future);

    @Query("SELECT t FROM Task t WHERE t.userId = :userId AND t.startTime >= :start AND t.startTime <= :end")
    List<Task> findTasksBetweenByUserId(@Param("userId") Long userId, @Param("start") LocalDateTime start, @Param("end") LocalDateTime end);

    // Count queries for user-specific statistics
    Long countByUserIdAndDone(Long userId, Boolean done);

    Long countByUserIdAndPriority(Long userId, String priority);

    Long countByUserIdAndCategory(Long userId, String category);

    @Query("SELECT count(t) FROM Task t WHERE t.userId = :userId AND t.done = false AND t.endTime < :currentTime")
    long countOverdueTasksByUserId(@Param("userId") Long userId, @Param("currentTime") LocalDateTime currentTime);

    @Query("SELECT count(t) FROM Task t WHERE t.userId = :userId AND t.startTime >= :start AND t.startTime <= :end")
    long countTasksBetweenByUserId(@Param("userId") Long userId, @Param("start") LocalDateTime start, @Param("end") LocalDateTime end);

    // Check if task exists and belongs to user
    boolean existsByIdAndUserId(Long id, Long userId);
}
