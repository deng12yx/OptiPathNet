import socket
import random
import time
import string


def generate_random_content(length):
    # 生成随机数据内容（英文字母和数字）
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def send_probe_packets(dst_ips, packet_size, num_packets, min_port, max_port):
    # 创建UDP套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 按顺序发送探测包到每个目标IP地址
    for i in range(num_packets):
        # 生成随机数据包内容
        udp_packet = generate_random_content(packet_size).encode()[:packet_size]
        dst_port = random.randint(min_port, max_port)
        src_port = random.randint(min_port, max_port)
        for dst_ip in dst_ips:
            # 发送数据包
            sock.sendto(udp_packet, (dst_ip, dst_port))
            # 等待一定时间，以达到发送速率的要求
            time.sleep(1 / (num_packets * packet_size * 8 / 1000))  # 发送速率单位为Kb/s
        if i % 100 == 0:
            print(f"Sent {i} packets")
        # 间隔 10 毫秒
        time.sleep(0.01)
    # 关闭套接字
    sock.close()


# 设置目标IP地址列表
dst_ips = ['10.0.0.14', '10.0.0.19']
# 设置数据包大小
packet_size = 46
# 设置发送的数据包数量
num_packets = int(100 * (1000 * 5 / 100) + 1)
# 设置端口号范围
min_port = 1024
max_port = 65535

# 发送探测包
send_probe_packets(dst_ips, packet_size, num_packets, min_port, max_port)
