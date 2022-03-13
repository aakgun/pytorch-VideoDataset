# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 13:07:11 2017

@author: IbrahimD
"""
import cv2
import math


class Balls:   
    
    ball_count = 0
    dict_balls = {}
    
    def __init__(self):
        self.ball_count = 0
        self.dict_balls = {}
    
    def draw(self,frame):
        ball_id = 1
        x = int(round(self.dict_balls[ball_id][0]))
        y = int(round(self.dict_balls[ball_id][1]))
        r = int(self.dict_balls[ball_id][2])
        
        cv2.circle(frame,(x, y), r, (0,255,0), -1)
        
    def add_ball(self,x,y,r,moving_angle):
        self.ball_count = self.ball_count + 1
        self.dict_balls[self.ball_count] = [x,y,r,moving_angle]
        
        
    def move_balls(self):
         ball_id = 1
         x = self.dict_balls[ball_id][0]
         y = self.dict_balls[ball_id][1]
         angle = self.dict_balls[ball_id][3]
         
         a = math.sin(math.radians(angle))
         b = math.cos(math.radians(angle))
         
         if angle>360:
             angle = 360-angle
           
#         print(x,y,angle)
        
         self.dict_balls[ball_id][0] = x + (20*b)
         self.dict_balls[ball_id][1] = y + (20*a) 
             
         if x<0:
             self.dict_balls[ball_id][3] = angle + 90
             self.dict_balls[ball_id][0] = 5

         elif x>640:
             self.dict_balls[ball_id][3] = angle + 90
             self.dict_balls[ball_id][0] = 630

         elif y<0:
             self.dict_balls[ball_id][3] = angle + 90
             self.dict_balls[ball_id][1] = 5
                            
         elif y>480:
             self.dict_balls[ball_id][3] = angle + 90                    
             self.dict_balls[ball_id][1] = 470
                            
           

         
                            
    def collision(self,kp):
         
        
         ball_id = 1
         x = self.dict_balls[ball_id][0]
         y = self.dict_balls[ball_id][1]
         r = self.dict_balls[ball_id][2]
         
         for point in kp:
             
            dist = math.sqrt( (point.pt[0] - x)**2 + (point.pt[1] - y)**2 )

            if dist < r:
                v1_theta = math.atan2(point.pt[0], x)
                v2_theta = math.atan2(point.pt[1], y)
                
                angle = (v2_theta - v1_theta) * (180.0 / math.pi)
                
                if angle < 0:
                    angle += 360.0
                
#                print(angle)
                self.dict_balls[ball_id][3] += angle + 90
                
                
                
                
                