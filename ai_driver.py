import torch
import numpy as np
import joblib
from train_model import DriverNN
import socket
import struct
import time
import sys

class AIDriver:
    def __init__(self, model_path='best_driver_model.pth', scaler_path='scaler.pkl'):
        # Load the model
        self.model = DriverNN(input_size=8)  # 8 features as defined in train_model.py
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set to evaluation mode
        
        # Load the scaler
        self.scaler = joblib.load(scaler_path)
        
        # Initialize socket connection
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        
        # Bot ID
        self.id = 'SCR'
        
        # Server address
        self.server_addr = ('localhost', 3001)
        
        # Control parameters
        self.steer_lock = 0.785398  # 45 degrees in radians
        self.steer_sensitivity = 0.7  # Increased steering sensitivity
        self.prev_rpm = None
        self.prev_gear = 1
        
    def init(self):
        """Return init string with rangefinder angles"""
        angles = [0 for x in range(19)]
        
        for i in range(5):
            angles[i] = -90 + i * 15
            angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            angles[i] = -20 + (i-5) * 5
            angles[18 - i] = 20 - (i-5) * 5
        
        return f"(init {angles})"
        
    def parse_sensor_data(self, buf_str):
        """Parse the sensor data string from TORCS"""
        try:
            # Extract values using string manipulation
            angle = float(buf_str.split('(angle ')[1].split(')')[0])
            track = float(buf_str.split('(track ')[1].split(')')[0].split()[0])  # Take first track sensor
            speedX = float(buf_str.split('(speedX ')[1].split(')')[0])
            speedY = float(buf_str.split('(speedY ')[1].split(')')[0])
            speedZ = float(buf_str.split('(speedZ ')[1].split(')')[0])
            trackPos = float(buf_str.split('(trackPos ')[1].split(')')[0])
            rpm = float(buf_str.split('(rpm ')[1].split(')')[0])
            gear = float(buf_str.split('(gear ')[1].split(')')[0])
            
            # Ensure gear is within valid range (1-6)
            gear = max(1, min(6, gear))
            
            return np.array([track, angle, speedX, speedY, speedZ, trackPos, rpm, gear])
        except Exception as e:
            print(f"Error parsing sensor data: {e}")
            print(f"Received data: {buf_str}")
            return None
    
    def format_control_command(self, steer, accel, gear):
        """Format the control command for TORCS"""
        # Apply steering sensitivity and lock
        steer = steer * self.steer_sensitivity * self.steer_lock
        
        # Ensure values are within valid ranges
        steer = max(-1.0, min(1.0, steer))
        accel = max(0.0, min(1.0, accel))
        gear = max(1, min(6, int(gear)))
        
        return f"(steer {steer:.3f})(accel {accel:.3f})(gear {gear})"
    
    def adjust_gear(self, rpm, current_gear):
        """Adjust gear based on RPM"""
        if self.prev_rpm is None:
            self.prev_rpm = rpm
            return current_gear
            
        # Determine if RPM is increasing
        rpm_increasing = (rpm - self.prev_rpm) > 0
        
        # Shift up if RPM is high and increasing
        if rpm_increasing and rpm > 7000:
            new_gear = min(6, current_gear + 1)
        # Shift down if RPM is low and decreasing
        elif not rpm_increasing and rpm < 3000:
            new_gear = max(1, current_gear - 1)
        else:
            new_gear = current_gear
            
        self.prev_rpm = rpm
        return new_gear
    
    def run(self):
        print("AI Driver started. Waiting for sensor data...")
        
        while True:
            # Send identification
            init_str = self.id + self.init()
            print(f"Sending init string: {init_str}")
            
            try:
                self.sock.sendto(init_str.encode(), self.server_addr)
            except socket.error as msg:
                print("Failed to send data...Exiting...")
                sys.exit(-1)
            
            try:
                buf, addr = self.sock.recvfrom(1000)
                buf_str = buf.decode()
                print(f"Received: {buf_str}")
                
                if buf_str.find("***identified***") >= 0:
                    print("Successfully identified with server")
                    break
            except socket.error as msg:
                print("Didn't get response from server...")
                continue
        
        while True:
            try:
                # Receive sensor data
                buf, addr = self.sock.recvfrom(1000)
                buf_str = buf.decode()
                
                if buf_str.find("***shutdown***") >= 0:
                    print("Server shutdown")
                    break
                
                if buf_str.find("***restart***") >= 0:
                    print("Server restart")
                    break
                
                # Parse sensor data
                sensor_data = self.parse_sensor_data(buf_str)
                if sensor_data is not None:
                    # Process sensor data
                    processed_data = self.process_sensor_data(sensor_data)
                    controls = self.get_control_outputs(processed_data)
                    
                    # Adjust gear based on RPM
                    rpm = sensor_data[6]  # RPM is the 7th element
                    controls['gear'] = self.adjust_gear(rpm, controls['gear'])
                    
                    # Format and send control command
                    control_str = self.format_control_command(
                        controls['steer'],
                        controls['accel'],
                        controls['gear']
                    )
                    
                    print(f"Sending control: {control_str}")
                    self.sock.sendto(control_str.encode(), self.server_addr)
                
            except socket.timeout:
                print("Waiting for data...")
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def process_sensor_data(self, sensor_data):
        # Normalize the sensor data using the same scaler used in training
        normalized_data = self.scaler.transform(sensor_data.reshape(1, -1))
        return torch.FloatTensor(normalized_data)
    
    def get_control_outputs(self, sensor_data):
        with torch.no_grad():
            outputs = self.model(sensor_data)
            
            # Split outputs into continuous and discrete parts
            continuous = outputs[0, :2]  # steering and acceleration
            gear_probs = outputs[0, 2:]  # gear probabilities (5 values)
            
            # Get the gear with highest probability (add 1 since gears are 1-5)
            gear = torch.argmax(gear_probs).item() + 1
            
            return {
                'steer': float(continuous[0]),
                'accel': float(continuous[1]),
                'gear': gear
            }

if __name__ == "__main__":
    driver = AIDriver()
    driver.run() 