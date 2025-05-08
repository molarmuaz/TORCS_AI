import sys
import argparse
import socket
import csv
import time
import driver
import numpy as np
import os
import threading

def create_socket(host_ip, host_port, max_retries=3):
    """Create a socket with improved error handling and reconnection logic"""
    for attempt in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.1)  # Reduced timeout for better responsiveness
            return sock
        except socket.error as e:
            print(f'Socket creation attempt {attempt + 1} failed: {e}')
            if attempt == max_retries - 1:
                raise
            time.sleep(1)

class TelemetryLogger:
    def __init__(self, logfile):
        self.logfile = logfile
        self.csv_file = None
        self.csv_writer = None
        self.rows_written = 0
        self.lock = threading.Lock()
        self.initialize_log_file()

    def initialize_log_file(self):
        try:
            # Ensure the log directory exists
            log_dir = os.path.dirname(self.logfile)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Open CSV file for telemetry logging
            self.csv_file = open(self.logfile, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header
            header = [
                'timestamp',
                'track_sensors',  # Track sensor readings (19 values)
                'speed',          # Current speed
                'accel',          # Acceleration input
                'brake',          # Brake input
                'steer',          # Steering input
                'gear',           # Current gear
                'rpm',            # Engine RPM
                'track_pos',      # Position on track (-1 to 1)
                'angle',          # Car angle
                'damage'          # Car damage
            ]
            self.csv_writer.writerow(header)
            self.csv_file.flush()
            print(f"Created log file: {self.logfile}")
        except Exception as e:
            print(f"Error creating log file: {e}")
            raise

    def log_telemetry(self, state, control):
        with self.lock:
            try:
                if self.csv_file is None or self.csv_file.closed:
                    self.initialize_log_file()

                timestamp = time.time()
                
                # Calculate speed from speedX, speedY, speedZ
                speed = np.sqrt(state.speedX**2 + state.speedY**2 + state.speedZ**2)
                
                # Prepare track sensors data (19 values)
                track_sensors = state.track if state.track else [0] * 19
                
                row = [
                    timestamp,
                    ','.join(map(str, track_sensors)),
                    speed,
                    control.accel,
                    control.brake,
                    control.steer,
                    state.gear,
                    state.rpm,
                    state.trackPos,
                    state.angle,
                    state.damage
                ]
                
                self.csv_writer.writerow(row)
                self.csv_file.flush()
                self.rows_written += 1
                
                if self.rows_written % 100 == 0:
                    print(f"Written {self.rows_written} rows to log file")
                    
            except Exception as e:
                print(f"Error writing to log file: {e}")
                # Try to recover the file
                try:
                    if self.csv_file is not None:
                        self.csv_file.close()
                    self.initialize_log_file()
                except Exception as recover_error:
                    print(f"Failed to recover log file: {recover_error}")

    def close(self):
        with self.lock:
            try:
                if self.csv_file is not None and not self.csv_file.closed:
                    self.csv_file.close()
                    print(f"Successfully closed log file. Total rows written: {self.rows_written}")
            except Exception as e:
                print(f"Error closing log file: {e}")

if __name__ == '__main__':
    # Configure argument parser
    parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')

    parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                        help='Host IP address (default: localhost)')
    parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                        help='Host port number (default: 3001)')
    parser.add_argument('--id', action='store', dest='id', default='SCR',
                        help='Bot ID (default: SCR)')
    parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                        help='Maximum number of learning episodes (default: 1)')
    parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                        help='Maximum number of steps (default: 0)')
    parser.add_argument('--track', action='store', dest='track', default=None,
                        help='Name of the track')
    parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                        help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
    parser.add_argument('--manual', action='store_true', dest='manual_mode', default=False,
                        help='Enable manual driving mode')
    parser.add_argument('--logfile', action='store', dest='logfile', default='telemetry_log.csv',
                        help='CSV file to log telemetry data (default: telemetry_log.csv)')

    arguments = parser.parse_args()

    # Print summary
    print(f'Connecting to server host IP: {arguments.host_ip} @ port: {arguments.host_port}')
    print(f'Bot ID: {arguments.id}')
    print(f'Maximum episodes: {arguments.max_episodes}')
    print(f'Maximum steps: {arguments.max_steps}')
    print(f'Track: {arguments.track}')
    print(f'Stage: {arguments.stage}')
    print(f'Manual Mode: {arguments.manual_mode}')
    print(f'Log File: {arguments.logfile}')
    print('*********************************************')

    try:
        sock = create_socket(arguments.host_ip, arguments.host_port)
    except socket.error as e:
        print('Could not create socket after multiple attempts.')
        sys.exit(-1)

    shutdownClient = False
    curEpisode = 0
    verbose = False

    d = driver.Driver(arguments.stage)
    d.manual_mode = arguments.manual_mode

    # Initialize telemetry logger
    logger = TelemetryLogger(arguments.logfile)

    while not shutdownClient:
        while True:
            print(f'Sending ID to server: {arguments.id}')
            buf = (arguments.id + d.init()).encode()

            print(f'Sending init string to server: {buf.decode()}')
            
            try:
                sock.sendto(buf, (arguments.host_ip, arguments.host_port))
            except socket.error as e:
                print(f"Failed to send data: {e}")
                continue
                
            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode()
            except socket.error:
                print("Didn't get response from server...")
                continue

            if '***identified***' in buf:
                print(f'Received: {buf}')
                break

        currentStep = 0
        last_log_time = time.time()
        log_interval = 0.05  # Log every 50ms

        while True:
            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode()
            except socket.error:
                print("Didn't get response from server...")
                continue
            
            if verbose:
                print(f'Received: {buf}')
            
            if '***shutdown***' in buf:
                d.onShutDown()
                shutdownClient = True
                print('Client Shutdown')
                break
            
            if '***restart***' in buf:
                d.onRestart()
                print('Client Restart')
                break
            
            currentStep += 1
            if currentStep != arguments.max_steps:
                if buf:
                    buf = d.drive(buf)
            else:
                buf = '(meta 1)'

            # Log telemetry data at regular intervals
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                logger.log_telemetry(d.state, d.control)
                last_log_time = current_time

            if verbose:
                print(f'Sending: {buf}')
            
            if buf:
                try:
                    sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
                except socket.error as e:
                    print(f"Failed to send data: {e}")
                    continue
        
        curEpisode += 1
        if curEpisode == arguments.max_episodes:
            shutdownClient = True

    # Clean up
    logger.close()
    sock.close()
