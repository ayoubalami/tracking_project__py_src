# import pigpio
# import time

# pi = pigpio.pi()

# servo_pin = 18
# initial_angle = 0
# final_angle = 180

# #Calculating the pulse width at initial angle
# pulse_width = 600 + (initial_angle * 10)
# pi.set_servo_pulsewidth(servo_pin, pulse_width)

# #setting the speed of rotation
# speed = 0.01 #in seconds

# #Moving the servo from initial to final angle
# # for angle in range(initial_angle,final_angle):
# #     pulse_width = 544 + (angle * 5)
# #     pi.set_servo_pulsewidth(servo_pin, pulse_width)
# #     time.sleep(speed)
# angle=0
# up=True
# while True:
   
#     if up :
#         angle+=1
#         if angle>180:
#             up=False
#     else:
#         angle-=1
#         if angle<0:
#             up=True 
    
    
#     pulse_width = 600 + (angle*2 )
#     pi.set_servo_pulsewidth(servo_pin, pulse_width)
#     time.sleep(0.05)
# pi.stop() 


import pigpio
import time

pi = pigpio.pi()

servo_pin = 18
initial_angle = 0
final_angle = 180

#Calculating the pulse width at initial angle
pulse_width = 600 + (initial_angle * 10)
pi.set_servo_pulsewidth(servo_pin, pulse_width)

# #setting the speed of rotation
# speed = 0.01 #in seconds

# #Moving the servo from initial to final angle
# # for angle in range(initial_angle,final_angle):
# #     pulse_width = 544 + (angle * 5)
# #     pi.set_servo_pulsewidth(servo_pin, pulse_width)
# #     time.sleep(speed)
# up=True
# while True:
   
#     if up :
#         pulse_width+=1
#         if pulse_width>2500:
#             pulse_width = 2500
#             up=False
#     else:
#         pulse_width-=1
#         if pulse_width<500:
#             pulse_width = 500
#             up=True 
    
#     pi.set_servo_pulsewidth(servo_pin, pulse_width)
#     time.sleep(0.0005)
# pi.stop() 

