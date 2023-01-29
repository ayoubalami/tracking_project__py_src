
import pigpio
import time

 
class ServoMotor:    

    def __init__(self,servo_pin):
        self.pi = pigpio.pi()
        self.servo_pin = servo_pin
        self.min_pulse=660
        self.max_pulse=2400 

    def goToAngleWithSpeed(self,angle,speed):
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
 
    def goToAngle(self,angle):
        pulse_width=(((self.max_pulse-self.min_pulse)*angle)/180)+self.min_pulse
        print(pulse_width)
        self.pi.set_servo_pulsewidth(self.servo_pin, pulse_width)
        
# motor=ServoMotor(servo_pin=23)
# # motor=ServoMotor(servo_pin=18) 
# # motor.goToAngle(angle=180) 
# # time.sleep(2)
# motor.goToAngle(angle=0) 
# time.sleep(1)

