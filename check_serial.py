import serial, time

s = serial.Serial('COM5', 115200, timeout=1)  # COM5 改成你的端口
t = time.time()
n = 0

while time.time() - t < 10:
    line = s.readline().decode('utf-8', 'ignore').strip()
    if line:
        n += 1
        print(line)

print('LINES=', n)
s.close()