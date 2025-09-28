package com.taskscheduler.taskservice.service;

import com.taskscheduler.taskservice.client.UserServiceClient;
import com.taskscheduler.taskservice.dto.UserDTO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

@Service
public class UserService {

    @Autowired
    private UserServiceClient userServiceClient;

    public boolean validateToken(String authHeader) {
        try {
            ResponseEntity<Boolean> response = userServiceClient.validateToken(authHeader);
            return response.getBody() != null && response.getBody();
        } catch (Exception e) {
            return false;
        }
    }

    public UserDTO getUserFromToken(String authHeader) throws Exception {
        ResponseEntity<Boolean> validationResponse = userServiceClient.validateToken(authHeader);
        if (validationResponse.getBody() == null || !validationResponse.getBody()) {
            throw new RuntimeException("Invalid token");
        }

        ResponseEntity<UserDTO> userResponse = userServiceClient.getUserInfo(authHeader);
        if (userResponse.getBody() == null) {
            throw new RuntimeException("Could not retrieve user information");
        }

        return userResponse.getBody();
    }
}