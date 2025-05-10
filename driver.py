import pygame
import msgParser
import carState
import carControl
import time

class Driver(object):
    '''
    A driver object for the SCRC.
    Manual control using pygame.
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage

        self.gear_up_pressed = False
        self.gear_down_pressed = False

        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        # Initialize manual control variables
        self.manual_steer = 0.0
        self.manual_accel = 0.0
        self.manual_gear = 1
        self.manual_brake = 0.0
        
        # Control parameters
        self.steer_speed = 0.015  # Significantly reduced for much less sensitive steering
        self.steer_return_speed = 0.008  # Gentler return to center
        self.accel_speed = 0.03  # Much gentler acceleration
        self.brake_speed = 0.03  # Much gentler braking
        self.last_update_time = time.time()
        self.control_update_interval = 1/120  # Reduced to 120 FPS for more stable control

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("TORCS Manual Control Input")
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])
        pygame.key.set_repeat(0)  # Disable key repeat

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for _ in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5
        
        return self.parser.stringify({'init': self.angles})

    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        if not pygame.get_init():
            pygame.init()
            self.screen = pygame.display.set_mode((400, 300))
            pygame.display.set_caption("TORCS Manual Control Input")
            pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])
            pygame.key.set_repeat(0)
        
        # Process all events to prevent input lag
        pygame.event.pump()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        keys = pygame.key.get_pressed()
        current_time = time.time()
        delta_time = min(current_time - self.last_update_time, 0.05)  # Reduced cap to 0.05s for faster updates
        
        # Only update controls at the target interval
        if delta_time >= self.control_update_interval:
            # Improved steering control with smoother transitions
            if keys[pygame.K_LEFT]:
                self.manual_steer = min(self.manual_steer + self.steer_speed, 1.0)
            elif keys[pygame.K_RIGHT]:
                self.manual_steer = max(self.manual_steer - self.steer_speed, -1.0)
            else:
                # Gradual return to center
                if self.manual_steer > 0:
                    self.manual_steer = max(0, self.manual_steer - self.steer_return_speed)
                elif self.manual_steer < 0:
                    self.manual_steer = min(0, self.manual_steer + self.steer_return_speed)

            # Improved acceleration control with smoother transitions
            if keys[pygame.K_UP]:
                self.manual_accel = min(self.manual_accel + self.accel_speed, 1.0)
                self.manual_brake = 0.0
            elif keys[pygame.K_DOWN]:
                self.manual_accel = 0.0
                self.manual_brake = min(self.manual_brake + self.brake_speed, 1.0)
            else:
                # Gradual deceleration
                self.manual_accel = max(0, self.manual_accel - self.accel_speed)
                self.manual_brake = max(0, self.manual_brake - self.brake_speed)

            if keys[pygame.K_SPACE]:
                self.manual_accel = 0.2
                self.manual_brake = 0.0

            # Gear control with debouncing
            if keys[pygame.K_a]:
                if not self.gear_down_pressed: 
                    self.manual_gear = max(-1, self.manual_gear - 1)
                    self.gear_down_pressed = True
            else:
                self.gear_down_pressed = False 

            if keys[pygame.K_d]:
                if not self.gear_up_pressed:  
                    self.manual_gear = min(6, self.manual_gear + 1)
                    self.gear_up_pressed = True
            else:
                self.gear_up_pressed = False  

            if keys[pygame.K_s]:
                self.manual_gear = 0
            if keys[pygame.K_r]:
                self.manual_gear = -1

            # Apply controls
            self.control.setSteer(self.manual_steer)
            self.control.setAccel(self.manual_accel)
            self.control.setBrake(self.manual_brake)
            self.control.setGear(self.manual_gear)
            
            self.last_update_time = current_time
        
        return self.control.toMsg()

    def onShutDown(self):
        pygame.quit()
    
    def onRestart(self):
        pass
