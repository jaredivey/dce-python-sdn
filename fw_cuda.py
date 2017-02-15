# Copyright (C) 2011 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Sample configuration file 
#
#[DEFAULT]
#
##wsapi_host=<hostip>
##wsapi_port=<port:8080>
##ofp_listen_host=<hostip>
#ofp_tcp_listen_port=6653
#observe_links=True
#explicit_drop=False
from __future__ import print_function
from __future__ import absolute_import

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet, arp, ipv4
from ryu.lib.packet import ether_types
import ryu.topology.api as api

import pycuda.driver as cuda
import pycuda.autoinit as autoinit  # noqa

import numpy

import time, cProfile, pstats, StringIO

class FwCudaSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(FwCudaSwitch13, self).__init__(*args, **kwargs)
        self.logger.info("%s: Starting app", time.time())

        self.first_pkt_in = 1
        self.next_array = []

        result_f = open("fw_kernel.ptx", "rb")
        self.result_data = result_f.read()
        result_f.close()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # install table-miss flow entry
        #
        # We specify NO BUFFER to max_len of the output action due to
        # OVS bug. At this moment, if we specify a lesser number, e.g.,
        # 128, OVS will send Packet-In with invalid buffer_id and
        # truncated packet data. In that case, we cannot output packets
        # correctly.  The bug has been fixed in OVS v2.1.0.
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    @set_ev_cls(event.EventLinkAdd, CONFIG_DISPATCHER)
    def new_link_handler(self, ev):
        pr,start = self.enableProf()

        links = api.get_all_link(self)
        switches = api.get_all_switch(self)

        self.next_array = self.FloydWarshall(switches, links)
        self.disableProf(pr,start,"FW")

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        
        # If you hit this you might want to increase
        # the "miss_send_length" of your switch
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes",
                              ev.msg.msg_len, ev.msg.total_len)
        msg = ev.msg
        datapath = msg.datapath
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # ignore lldp packet
            return
        pr,start = self.enableProf()
        
        # Figure out environment
        links = api.get_all_link(self)
        switches = api.get_all_switch(self)
        hosts = api.get_all_host(self)

        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            arp_pkt = pkt.get_protocols(arp.arp)[0]
            if arp_pkt.opcode == arp.ARP_REQUEST:                
                # Send to ARP proxy. Cannot perform NIx routing until both hosts
                # are known by the controller
                self.ArpProxy (msg.data, datapath, in_port, links, switches, hosts)
            elif arp_pkt.opcode == arp.ARP_REPLY:
                self.ArpReply (msg.data, datapath, arp_pkt.dst_ip, links, switches, hosts)
            self.disableProf(pr,start,"ARP")
            return

        #self.logger.info("%s: packet in %s %s %s %s", time.time(), datapath.id, eth.src, eth.dst, in_port)
        
        # Start nix vector code
        src_ip = ''
        dst_ip = ''
        srcNode = ''
        dstNode = ''
        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            arp_pkt = pkt.get_protocols(arp.arp)[0]
            src_ip = arp_pkt.src_ip
            dst_ip = arp_pkt.dst_ip
        elif eth.ethertype == ether_types.ETH_TYPE_IP:
            ipv4_pkt = pkt.get_protocols(ipv4.ipv4)[0]
            src_ip = ipv4_pkt.src
            dst_ip = ipv4_pkt.dst
        for host in hosts:
            if src_ip == host.ipv4[0]:
                srcNode = host
            if dst_ip == host.ipv4[0]:
                dstNode = host

        srcSwitch = [switch for switch in switches if switch.dp.id == srcNode.port.dpid][0]
        dstSwitch = [switch for switch in switches if switch.dp.id == dstNode.port.dpid][0]

        sdnNix = []
        self.BuildNixVector (srcSwitch, dstSwitch, links, sdnNix)

        # Need to send to last switch to send out host port
        sdnNix.append((dstSwitch, dstNode.port.port_no))
        
        for curNix in sdnNix:
            self.sendNixRules (srcSwitch, curNix[0], curNix[1], msg)
        self.disableProf(pr,start,"COMPLETION")
        
    def ArpProxy(self, data, datapath, in_port, links, switches, hosts):
        for switch in switches:
            # Get all usable ports for this switch and then remove those ports
            # associated with switch-to-switch connections to look at edge ports
            all_ports = set([port.port_no for port in switch.ports if port.is_live()])
            src_ports = set([link.src.port_no for link in links if link.src.dpid == switch.dp.id and link.src.is_live()])
            dst_ports = set([link.dst.port_no for link in links if link.dst.dpid == switch.dp.id and link.dst.is_live()])
            link_ports = src_ports | dst_ports
            non_link_ports = list(all_ports - link_ports)
            
            # Push an ARP request out of the network edges
            for port in non_link_ports:
                if switch.dp != datapath or port != in_port:
                    actions = [switch.dp.ofproto_parser.OFPActionOutput(port)]
                    out = switch.dp.ofproto_parser.OFPPacketOut(datapath=switch.dp,
                                              buffer_id=switch.dp.ofproto.OFP_NO_BUFFER,
                                              in_port=switch.dp.ofproto.OFPP_CONTROLLER,
                                              actions=actions, data=data)
                    
                    self.logger.info("%s: Sending ARP Request: dpid=%s, port=%s", time.time(), switch.dp.id, port)
                    switch.dp.send_msg(out)

    def ArpReply(self, data, datapath, dst_ip, links, switches, hosts):
        for host in hosts:
            for switch in switches:
                # Push an ARP reply out of the appropriate switch port
                if host.port.dpid == switch.dp.id and host.ipv4.count(dst_ip):
                    actions = [switch.dp.ofproto_parser.OFPActionOutput(host.port.port_no)]
                    out = switch.dp.ofproto_parser.OFPPacketOut(datapath=switch.dp,
                                              buffer_id=switch.dp.ofproto.OFP_NO_BUFFER,
                                              in_port=switch.dp.ofproto.OFPP_CONTROLLER,
                                              actions=actions, data=data)

                    self.logger.info("%s: Sending ARP Reply: dpid=%s, ip=%s, port=%s", time.time(), switch.dp.id, dst_ip, host.port.port_no)
                    switch.dp.send_msg(out)
    
    def BuildNixVector (self, srcSwitch, dstSwitch, links, sdnNix):
        if srcSwitch == dstSwitch:
            return True

        src = numpy.int32(srcSwitch.dp.id)
        dst = numpy.int32(dstSwitch.dp.id)
        N = numpy.int32(max(self.next_array) + 1)
        if self.next_array[src * N + dst] == None:
            return False

        while src != dst:
            currSwitch = api.get_switch(self, src)[0]
            for link in links:
                if link.src.dpid == currSwitch.dp.id and link.dst.dpid == self.next_array[src][dst]:
                    sdnNix.append((currSwitch, link.src.port_no))
                    src = self.next_array[src][dst]
                    break

        return True

    def FloydWarshall(self, switches, links):
        adj_graph = dict()
        for switch1 in switches:
            adj_graph[switch1.dp.id] = dict()
            adj_graph[switch1.dp.id][switch1.dp.id] = 0
            for link in links:
                if link.src.dpid == switch1.dp.id:
                    adj_graph[switch1.dp.id][link.dst.dpid] = float(link.delay)

        N=max(adj_graph)+1
        adj_array = numpy.full(N*N, float("inf")).astype(numpy.float32)
        for key1, row in adj_graph.iteritems():
            for key2, value in row.iteritems():
                adj_array[key1 * N + key2] = value

        adj_gpu = cuda.mem_alloc(adj_array.size * adj_array.dtype.itemsize)
        cuda.memcpy_htod(adj_gpu, adj_array)

        next_array = [ i % N for i in range(N*N) ]
        next_np = numpy.array(next_array).astype(numpy.int32)
        next_gpu = cuda.mem_alloc(next_np.size * next_np.dtype.itemsize)
        cuda.memcpy_htod(next_gpu, next_np)

        mod = cuda.module_from_buffer(self.result_data)
        func = mod.get_function("fw")
        for k in range(1,N):
            func(adj_gpu, next_gpu, numpy.int32(k), numpy.int32(N), block=(N, N, 1), grid=(1, 1), shared=0)

        cuda.memcpy_dtoh(next_np, next_gpu)
        #cuda.memcpy_dtoh(adj_array, adj_gpu)

        next_gpu.free()
        adj_gpu.free()

        autoinit.patch_finish()

        #self.logger.info("%s", adj_array)
        #self.logger.info("%s", next_np)
        return next_np

    def sendNixRules(self, srcSwitch, switch, port_no, msg):
        ofproto = switch.dp.ofproto
        parser = switch.dp.ofproto_parser

        actions = [switch.dp.ofproto_parser.OFPActionOutput(port_no)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]

        match = parser.OFPMatch(eth_src=msg.match['eth_src'],
                                eth_dst=msg.match['eth_dst'])
        mod = parser.OFPFlowMod(datapath=switch.dp, priority=1,
                                match=match, instructions=inst)
        switch.dp.send_msg(mod)

        #self.logger.info("%s: Sending Nix rule: dpid=%s, port=%s", time.time(), switch.dp.id, port_no)
        if srcSwitch.dp.id == switch.dp.id:
            data = None

            if msg.buffer_id == ofproto.OFP_NO_BUFFER:
                data = msg.data

            out = parser.OFPPacketOut(datapath=srcSwitch.dp, buffer_id=msg.buffer_id,
                                      in_port=switch.dp.ofproto.OFPP_CONTROLLER,
                                      actions=actions, data=data)
            srcSwitch.dp.send_msg(out)

    def bin(self, s):
        return str(s) if s<=1 else bin(s>>1) + str(s&1)

    def enableProf(self):
        pr = cProfile.Profile()
        pr.enable()
        return pr,time.time()

    def disableProf(self, pr, start, whichcase):
        completion = time.time() - start
        pr.disable()
        s = StringIO.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.print_stats(0)
        self.logger.info("%s\t%f\t%s", whichcase, completion, s.getvalue())
