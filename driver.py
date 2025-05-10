import msgParser
import carState
import carControl
import keyboard
import csv
import time
from datetime import datetime

class Driver(object):
    '''
    A driver object for the SCRC that combines manual and AI control
    '''

    def __init__(self, stage, ai_mode=False):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        self.ai_mode = ai_mode
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        # Control parameters
        self.steer_lock = 0.785398  # 45 degrees in radians
        self.max_speed = 100
        self.prev_rpm = None
        
        # Steering sensitivity adjustment
        self.manual_steer_sensitivity = 0.5  # Reduce manual steering sensitivity
        self.ai_steer_sensitivity = 0.2  # Reduce AI steering sensitivity
        
        # Initialize with proper gear (1) and zero controls
        self.control.setGear(1)
        self.control.setAccel(0.0)
        self.control.setBrake(0.0)
        self.control.setSteer(0.0)
        
        # Setup logging with the new format
        self.log_file = f"car_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.setup_logging()
        
        if not self.ai_mode:
            self.setup_keyboard_controls()
            print("Manual control mode active. Use WASD keys to drive.")
        else:
            print("AI control mode active.")

    def setup_keyboard_controls(self):
        """Setup keyboard controls for manual driving"""
        keyboard.unhook_all()
        
        # Acceleration control
        keyboard.on_press_key('w', lambda _: self.set_accel(1.0))
        keyboard.on_release_key('w', lambda _: self.set_accel(0.0))
        
        # Brake control
        keyboard.on_press_key('s', lambda _: self.set_brake(1.0))
        keyboard.on_release_key('s', lambda _: self.set_brake(0.0))
        
        # Steering controls - reduced sensitivity
        keyboard.on_press_key('a', lambda _: self.set_steer(1.0))  # Left
        keyboard.on_release_key('a', lambda _: self.set_steer(0.0))
        
        keyboard.on_press_key('d', lambda _: self.set_steer(-1.0))  # Right
        keyboard.on_release_key('d', lambda _: self.set_steer(0.0))
        
        # Gear shifting
        keyboard.on_press_key('q', lambda _: self.shift_gear(-1))
        keyboard.on_press_key('e', lambda _: self.shift_gear(1))
        
        # Mode toggle
        keyboard.on_press_key('m', lambda _: self.toggle_mode())

    def set_accel(self, value):
        """Control acceleration"""
        if self.state.getGear() != 0:  # Only accelerate if not in neutral
            self.control.setAccel(value)
            if value > 0:
                self.control.setBrake(0.0)

    def set_brake(self, value):
        """Control braking"""
        self.control.setBrake(value)
        if value > 0:
            self.control.setAccel(0.0)

    def set_steer(self, value):
        """Control steering with reduced sensitivity"""
        # Apply sensitivity adjustment to manual steering
        adjusted_value = value * self.manual_steer_sensitivity
        self.control.setSteer(adjusted_value * self.steer_lock)

    def shift_gear(self, direction):
        """Shift gears up or down"""
        current_gear = self.state.getGear() or 1
        new_gear = current_gear + direction
        if 1 <= new_gear <= 6:  # Only allow valid gears
            self.control.setGear(new_gear)

    def toggle_mode(self):
        """Toggle between AI and manual control"""
        self.ai_mode = not self.ai_mode
        print(f"Control mode: {'AI' if self.ai_mode else 'Manual'}")
        # Reset controls when switching modes
        self.control.setGear(1)
        self.control.setAccel(0.0)
        self.control.setBrake(0.0)
        self.control.setSteer(0.0)

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})

    def drive(self, msg):
        """Main driving loop"""
        # First parse the message to update car state
        self.state.setFromMsg(msg)
        
        # Ensure we have valid gear
        if self.state.getGear() == 0:
            self.control.setGear(1)
        
        if self.ai_mode:
            # AI control logic
            self.steer()
            self.gear()
            self.speed()
        
        # Log data regardless of mode
        self.log_data()
        
        return self.control.toMsg()

    def steer(self):
        """AI steering logic with improved sensitivity"""
        angle = self.state.angle
        dist = self.state.trackPos
        
        # Apply AI steering sensitivity and smooth out steering
        # Reduce the impact of trackPos and apply sensitivity adjustment
        steer_value = (angle - dist * 0.3) * self.ai_steer_sensitivity
        
        # Apply rate limiting to prevent sudden steering changes
        current_steer = self.control.getSteer()
        if current_steer is not None:
            # Limit how quickly steering can change to smooth out control
            max_change = 0.1
            if steer_value > current_steer + max_change:
                steer_value = current_steer + max_change
            elif steer_value < current_steer - max_change:
                steer_value = current_steer - max_change
        
        self.control.setSteer(steer_value)
    
    def gear(self):
        """AI gear shifting logic"""
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        
        if self.prev_rpm == None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False
        
        if up and rpm > 7000:
            gear += 1
        
        if not up and rpm < 3000:
            gear -= 1
        
        # Ensure gear is within valid range
        gear = max(1, min(6, gear))
        
        self.control.setGear(gear)
        self.prev_rpm = rpm
    
    def speed(self):
        """AI speed control logic"""
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        
        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
            self.control.setBrake(0.0)
        else:
            accel = 0.0
            self.control.setBrake(0.1)
        
        self.control.setAccel(accel)

    def setup_logging(self):
        """Setup CSV logging with the new format"""
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Acceleration', 'Brake', 'Gear', 'Steer', 'Clutch', 'Focus', 'Meta',
                'Angle', 'CurLapTime', 'Damage', 'DistFromStart', 'DistRaced',
                'Focus2', 'Fuel', 'Gear2', 'LastLapTime', 'Opponents', 'RacePos',
                'RPM', 'Speed X', 'Speed Y', 'Speed Z', 'Track', 'TrackPos',
                'WheelSpinVel', 'Z'
            ])

    def log_data(self):
        """Log car data to CSV with the new format"""
        try:
            with open(self.log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Format track data as space-separated string
                track_str = " ".join([str(t) for t in self.state.getTrack()]) if self.state.getTrack() else ""
                
                # Format opponents data as space-separated string
                opponents_str = " ".join([str(o) for o in self.state.getOpponents()]) if self.state.getOpponents() else ""
                
                # Format wheel spin velocity data as space-separated string
                wheelspinvel_str = " ".join([str(w) for w in self.state.getWheelSpinVel()]) if self.state.getWheelSpinVel() else ""
                
                # Format focus data as space-separated string
                focus_str = " ".join([str(f) for f in self.state.focus]) if self.state.focus else ""
                
                writer.writerow([
                    self.control.getAccel() or 0,
                    self.control.getBrake() or 0,
                    self.control.getGear() or 1,
                    self.control.getSteer() or 0,
                    self.control.getClutch() or 0,
                    self.control.focus or 0,
                    self.control.getMeta() or 0,
                    self.state.getAngle() or 0,
                    self.state.getCurLapTime() or 0,
                    self.state.getDamage() or 0,
                    self.state.getDistFromStart() or 0,
                    self.state.getDistRaced() or 0,
                    focus_str,
                    self.state.getFuel() or 0,
                    self.state.getGear() or 0,
                    self.state.lastLapTime or 0,
                    opponents_str,
                    self.state.getRacePos() or 0,
                    self.state.getRpm() or 0,
                    self.state.getSpeedX() or 0,
                    self.state.getSpeedY() or 0,
                    self.state.getSpeedZ() or 0,
                    track_str,
                    self.state.getTrackPos() or 0,
                    wheelspinvel_str,
                    self.state.getZ() or 0
                ])
        except Exception as e:
            print(f"Error logging data: {e}")
            
    def onShutDown(self):
        """Handle shutdown event"""
        keyboard.unhook_all()
        print("Driver shutdown")
    
    def onRestart(self):
        """Handle restart event"""
        self.control.setGear(1)
        self.control.setAccel(0.0)
        self.control.setBrake(0.0)
        self.control.setSteer(0.0)
        print("Driver restarted")