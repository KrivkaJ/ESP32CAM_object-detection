import serial
import time

port = 'COM4'
baund_rate = 115200
esp = serial.Serial(port,baund_rate,timeout=1)
def send_data(nunmber):
    command = f"{nunmber}\n"
    esp.write(command.encode('utf-8'))
send_data(1)
time.sleep(5)
send_data(2)
time.sleep(5)
send_data(1)
time.sleep(5)
send_data(2)
time.sleep(5)