import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 5005))
print("listening...")
while True:
    data, addr = s.recvfrom(512)
    print(f"{addr} -> {len(data)}B, magic={data[0]:02x}")