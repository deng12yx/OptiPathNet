import json

import numpy as np
from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick

from ryu.ofproto import ofproto_v1_3

from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, DEAD_DISPATCHER, \
    HANDSHAKE_DISPATCHER  #只是表示datapath数据路径的状态
from ryu.controller.handler import set_ev_cls

from ryu.lib import hub
from ryu.lib.packet import packet, ethernet

from ryu.topology.switches import Switches
from ryu.topology.switches import LLDPPacket

import time
from bottleLink import get_all_CBR
from maxDelay import delay_needAdd, newDelayCompute, calculate_growth_rate, calculate_decline_rate, \
    generate_max_weight_edges

ECHO_REQUEST_INTERVAL = 0.05
DELAY_DETECTING_PERIOD = 5


class DelayDetect(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DelayDetect, self).__init__(*args, **kwargs)
        self.name = "delay"

        self.topology = lookup_service_brick(
            "topology")  #注意：我们使用lookup_service_brick加载模块实例时，对于我们自己定义的app,我们需要在类中定义self.name。
        self.switches = lookup_service_brick(
            "switches")  #此外，最重要的是：我们启动本模块DelayDetect时，必须同时启动自定义的模块！！！ 比如：ryu-manager ./TopoDetect.py ./DelayDetect.py --verbose --observe-links

        self.dpid2switch = {}  #或者直接为{}，也可以。下面_state_change_handler也会添加进去
        self.dpid2echoDelay = {}  #记录echo时延

        self.src_sport_dst2Delay = {}  #记录LLDP报文测量的时延。实际上可以直接更新，这里单独记录，为了单独展示 {”src_dpid-srt_port-dst_dpid“：delay}

        self.delayinfo = {}
        self.detector_thread = hub.spawn(self._detector)
        self.delayArray = []
        self.newDelayArray = []
        #self.CBR = self.topology.CBR

    def _detector(self):
        """
        协程实现伪并发，探测链路时延
        """
        while True:

            print(f"state is {self.topology.net_flag}")
            if self.topology == None:
                self.topology = lookup_service_brick("topology")
            if self.topology.net_flag:
                #print("------------------_detector------------------")
                self._send_echo_request()
                self.get_link_delay()
                if self.topology.net_flag:
                    self.show_delay()
                    #self.topology.show_topology()

            hub.sleep(DELAY_DETECTING_PERIOD)  #5秒一次

    def _send_echo_request(self):
        """
        发生echo报文到datapath
        """
        for datapath in self.dpid2switch.values():
            parser = datapath.ofproto_parser
            echo_req = parser.OFPEchoRequest(datapath, data=bytes("%.12f" % time.time(), encoding="utf8"))  #获取当前时间

            datapath.send_msg(echo_req)

            #重要！不要同时发送echo请求，因为它几乎同时会生成大量echo回复。
            #在echo_reply_处理程序中处理echo reply时，会产生大量队列等待延迟。
            hub.sleep(ECHO_REQUEST_INTERVAL)

    @set_ev_cls(ofp_event.EventOFPEchoReply, [MAIN_DISPATCHER, CONFIG_DISPATCHER, HANDSHAKE_DISPATCHER])
    def echo_reply_handler(self, ev):
        """
        处理echo响应报文，获取控制器到交换机的链路往返时延
              Controller
                  |
     echo latency |
                 `|‘
                   Switch
        """
        now_timestamp = time.time()
        try:
            echo_delay = now_timestamp - eval(ev.msg.data)
            self.dpid2echoDelay[ev.msg.datapath.id] = echo_delay
        except:
            return

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):  #处理到达的LLDP报文，从而获得LLDP时延
        """
                      Controller
                    |        /|\
                   \|/         |
                Switch----->Switch
        """
        msg = ev.msg

        try:
            src_dpid, src_outport = LLDPPacket.lldp_parse(msg.data)  #获取两个相邻交换机的源交换机dpid和port_no(与目的交换机相连的端口)
            dst_dpid = msg.datapath.id  #获取目的交换机（第二个），因为来到控制器的消息是由第二个（目的）交换机上传过来的
            dst_inport = msg.match['in_port']
            if self.switches is None:
                self.switches = lookup_service_brick("switches")  #获取交换机模块实例

            #获得key（Port类实例）和data（PortData类实例）
            for port in self.switches.ports.keys():  #开始获取对应交换机端口的发送时间戳
                if src_dpid == port.dpid and src_outport == port.port_no:  #匹配key
                    port_data = self.switches.ports[port]  #获取满足key条件的values值PortData实例，内部保存了发送LLDP报文时的timestamp信息
                    timestamp = port_data.timestamp
                    if timestamp:
                        delay = time.time() - timestamp
                        self._save_delay_data(src=src_dpid, dst=dst_dpid, src_port=src_outport, lldpdealy=delay)
        except:
            return

    def _save_delay_data(self, src, dst, src_port, lldpdealy):
        key = "%s-%s-%s" % (src, src_port, dst)
        self.src_sport_dst2Delay[key] = lldpdealy

    def get_link_delay(self):
        """
        更新图中的权值信息
        """
        print("--------------get_link_delay-----------")
        for src_sport_dst in self.src_sport_dst2Delay.keys():
            src, sport, dst = tuple(map(eval, src_sport_dst.split("-")))
            if src in self.dpid2echoDelay.keys() and dst in self.dpid2echoDelay.keys():
                sid, did = self.topology.dpid2id[src], self.topology.dpid2id[dst]
                #print("sid is ",sid)
                #print("did is",did)
                #if self.topology.net_topo[sid][did] == 0:

            s_d_delay = self.src_sport_dst2Delay[src_sport_dst] - (
                    self.dpid2echoDelay[src] + self.dpid2echoDelay[dst]) / 2;
            if src not in self.delayinfo.keys():
                self.delayinfo[src] = {}
            self.delayinfo[src][dst] = max(s_d_delay, 0)
            if s_d_delay < 0:  #注意：可能出现单向计算时延导致最后小于0，这是不允许的。则不进行更新，使用上一次原始值
                self.topology.net_topo[sid][did][1] = 0

            else:
                self.topology.net_topo[sid][did][1] = s_d_delay

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if not datapath.id in self.dpid2switch:
                self.logger.debug('Register datapath: %016x', datapath.id)
                self.dpid2switch[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.dpid2switch:
                self.logger.debug('Unregister datapath: %016x', datapath.id)
                del self.dpid2switch[datapath.id]

        if self.topology == None:
            self.topology = lookup_service_brick("topology")
        print("-----------------------_state_change_handler-----------------------")
        #print(self.topology.show_topology())
        print(self.switches)

    def show_delay(self):
        print("-----------------------show echo delay-----------------------")
        #for key, val in self.dpid2echoDelay.items():
        #    print("s%d----%.12f" % (key, val))
        #print("-----------------------show LLDP delay-----------------------")
        #for key, val in self.src_sport_dst2Delay.items():
        #    print("%s----%.12f" % (key, val))
        print("-----------------------show link delay-----------------------")
        self.bottleLinkCompute()

    def bottleLinkCompute(self):
        """
        计算瓶颈链路
        """
        #CBR = get_all_CBR()
        """
        更新图中的权值信息
        """
        weight_info = {}
        delayPre = []
        delayPredict = {}
        delayLater = {}
        for src_sport_dst in self.src_sport_dst2Delay.keys():
            src, sport, dst = tuple(map(eval, src_sport_dst.split("-")))
            if src in self.dpid2echoDelay.keys() and dst in self.dpid2echoDelay.keys():
                sid, did = self.topology.dpid2id[src], self.topology.dpid2id[dst]
                #res = float(CBR[src][sport]) * 8 + (self.delayinfo[src][dst] + self.delayinfo[dst][src]) / 2 * 100
                res = (self.delayinfo[src][dst] + self.delayinfo[dst][src]) / 2 * 100
                delayPre.append(self.delayinfo[src][dst])
                delayPre.append(self.delayinfo[dst][src])
                delayPredict[(src, dst)] = self.delayinfo[src][dst]
                #print("res is :", res)
                weight_info[(src, dst)] = res
        # 提取权重值形成列表
        weight_values = delayPre
        # 计算权重的平均值
        mean_weight = np.mean(weight_values)
        # 计算每个权重与平均值之间的差的平方
        squared_diff = [(w - mean_weight) ** 2 for w in weight_values]
        # 计算方差
        variance = np.mean(squared_diff)
        self.delayArray.append(delayPredict)
        start_time = time.time()
        #print(f"weight_info is {weight_info}")
        max_weight = generate_max_weight_edges(weight_info)
        #print(f"max_delays is {max_delays},max_delay_links is {max_delay_links}")
        new_delay = newDelayCompute(max_weight, self.delayinfo)
        for item, inner_dict in new_delay.items():
            for i, value in inner_dict.items():
                delayLater[(item, i)] = value
        self.newDelayArray.append(delayLater)
        # 结束记录时间
        end_time = time.time()
        # 计算运行时间
        run_time = end_time - start_time
        print(f"延迟策略计算时间：{run_time}")
        if len(self.delayArray) % 50 == 0:
            # print(f"delayArray is {self.delayArray}")
            # print(f"newDelayArray is {self.newDelayArray}")
            with open('newDelayArray.txt', 'w') as file1, open('delayArray.txt', 'w') as file2:
                file1.write(str(self.newDelayArray))
                file2.write(str(self.delayArray))
            print("文件写入成功")
            print("权重的平均值：", mean_weight)
            print("方差：", variance)
            #print(f"延迟策略计算时间：{run_time}")
            delays = []
            # 将所有节点的延迟值提取出来
            for node in new_delay.values():
                delays.extend(list(node.values()))
            # 计算延迟值的平均值
            mean_delay = np.mean(delays)

            # 计算平方差的累加
            squared_diff_sum = np.sum([(delay - mean_delay) ** 2 for delay in delays])
            # 计算方差
            variance2 = squared_diff_sum / len(delays)
            print("new权重的平均值：", mean_delay)
            print("new方差：", variance2)
            growth_rate = calculate_growth_rate(mean_weight, mean_delay)
            decline_rate = calculate_decline_rate(variance, variance2)
            print("增长率：", growth_rate)
            print("下降率：", decline_rate)
