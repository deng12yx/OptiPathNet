import urllib.request
import json


def get_all_ports(dpid):
    url = f"http://127.0.0.1:8080/stats/port/{dpid}"
    with urllib.request.urlopen(url) as response:
        data = response.read().decode('utf-8')
        res = json.loads(data)
        return res


def get_all_switches(dpid):
    url = f"http://127.0.0.1:8080/stats/flow/{dpid}"
    with urllib.request.urlopen(url) as response:
        data = response.read().decode('utf-8')
        res = json.loads(data)
        return res


def get_all_swithches_id():
    url = f"http://127.0.0.1:8080/stats/switches"
    with urllib.request.urlopen(url) as response:
        data = response.read().decode('utf-8')
        res = json.loads(data)
        return res


def get_all_CBR():
    datapathId = get_all_swithches_id()
    print(f"datapathId is {datapathId}")
    CBR = {}
    # 统计数据并分析
    for dpid in datapathId:
        CBR[dpid] = {}
        port_stats = get_all_ports(dpid=dpid)[f'{dpid}']  # 提取port为1的数据
        for stat in port_stats:
            tx_packets = stat['tx_packets']
            rx_packets = stat['rx_packets']
            duration_sec = stat['duration_sec']
            sum_packets = tx_packets + rx_packets
            CBR_single = sum_packets / duration_sec
            CBR[int(dpid)][int(stat['port_no'])] = CBR_single

    return CBR


if __name__ == '__main__':
    CBR = get_all_CBR()
    print("-----------------------show CBR-----------------------")
    for dpid, port_stats in CBR.items():
        print(f"dpid: {dpid}")
        for port, CBR in port_stats.items():
            print(f"port: {port}, CBR: {CBR}")
