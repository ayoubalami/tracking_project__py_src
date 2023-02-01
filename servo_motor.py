
import pigpio
import time

class ServoMotor:    

    def __init__(self,servo_pin):
        self.pi = pigpio.pi()
        self.servo_pin = servo_pin
        self.min_pulse=660
        self.max_pulse=2400 
        self.target_position=None

    def goToAngleWithSpeed(self,angle,speed=0.005):
        angle=angle+90
        target_pulse_width=round(((self.max_pulse-self.min_pulse)*angle)/180)+self.min_pulse
        print(target_pulse_width)
        current_pulse_width = self.pi.get_servo_pulsewidth(self.servo_pin)
        print("Current pulse width: ", current_pulse_width)
        while(current_pulse_width!=target_pulse_width):
            if (current_pulse_width>target_pulse_width):
                current_pulse_width-=1
            else:
                current_pulse_width+=1
            self.pi.set_servo_pulsewidth(self.servo_pin, current_pulse_width)
            time.sleep(speed)
        print("Done")

    def addAngleWithSpeed(self,angle_to_add,speed=0.01):

        current_pulse_width = self.pi.get_servo_pulsewidth(self.servo_pin)
        target_pulse_width=current_pulse_width+round(((self.max_pulse-self.min_pulse)*angle_to_add)/180)
        # print(target_pulse_width)
        print("Current pulse width: ", current_pulse_width)
        print("target_pulse_width pulse width: ", target_pulse_width)
        while(current_pulse_width!=target_pulse_width):
            if (current_pulse_width>target_pulse_width):
                current_pulse_width-=1
            else:
                current_pulse_width+=1
            self.pi.set_servo_pulsewidth(self.servo_pin, current_pulse_width)
            time.sleep(speed)
        print("Done")
 
    def goToAngle(self,angle):
        angle=angle+90
        pulse_width=(((self.max_pulse-self.min_pulse)*angle)/180)+self.min_pulse
        print(pulse_width)
        self.pi.set_servo_pulsewidth(self.servo_pin, pulse_width)
         

    def trackTargetObject(self,origins,destination_coordinates,speed=0.005):
        
        epsilon=2
        (width,heigth)=origins
        (dest_x,dest_y)=destination_coordinates
        centerX=int(width/2)
        centerY=int(heigth/2)
        # while(abs(centerX-dest_x)>epsilon):
        if (centerX-dest_x)>epsilon:
            self.x_servo_motor.addAngleWithSpeed(angle_to_add=1,speed=0.005) 
        elif (centerX-dest_x)<epsilon :
            self.x_servo_motor.addAngleWithSpeed(angle_to_add=-1,speed=0.005) 

        # # while(abs(centerX-dest_x)>epsilon):
        # i=0
        # while(i<50):
        #     print(centerX)
        #     print(dest_x)
        #     print("====")
        #     print(centerX-dest_x)
        #     if centerX>dest_x:
        #         self.x_servo_motor.moveRight()
        #     else:
        #         self.x_servo_motor.moveLeft()
        #     i+=1
        #     time.sleep(0.01)

        print(centerX-dest_x)
        print(centerY-dest_y)



        
#  def moveRight(self):
#         current_pulse_width = self.pi.get_servo_pulsewidth(self.servo_pin)
#         if direction=='R':
#             destination_pulse=current_pulse_width+1
#         if direction=='L':
#             destination_pulse=current_pulse_width-1
#         self.pi.set_servo_pulsewidth(self.servo_pin, destination_pulse)

# motor=ServoMotor(servo_pin=23)
# # motor=ServoMotor(servo_pin=18) 
# motor.goToAngle(angle=180) 
# # time.sleep(2)
# motor.goToAngle(angle=0) 
# time.sleep(1)

