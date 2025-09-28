package com.taskscheduler.taskservice.client;

import com.taskscheduler.taskservice.dto.UserDTO;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestHeader;

@FeignClient(name = "user-service")
public interface UserServiceClient {

    @PostMapping("/api/auth/validate")
    ResponseEntity<Boolean> validateToken(@RequestHeader("Authorization") String token);

    @PostMapping("/api/auth/user-info")
    ResponseEntity<UserDTO> getUserInfo(@RequestHeader("Authorization") String token);
}