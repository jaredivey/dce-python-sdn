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

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import mac
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet, arp, ipv4
from ryu.lib.packet import ether_types
import ryu.topology.api as api

import time

UINT64_MAX = (1 << 64) - 1

class NixMpls13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(NixMpls13, self).__init__(*args, **kwargs)
        self.logger.info("%s: Starting app", time.time())

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

    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, CONFIG_DISPATCHER)
    def port_desc_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        for port in datapath.ports:
            if port != ofproto.OFPP_LOCAL:
                    self.logger.info("%s has port %s", datapath, port)
                    mpls_match = parser.OFPMatch()
                    mpls_match.append_field(ofproto.OXM_OF_ETH_TYPE, ether_types.ETH_TYPE_MPLS)
                    mpls_match.append_field(ofproto.OXM_OF_MPLS_LABEL, port)
                    mpls_actions = [parser.OFPActionPopMpls(),
                                    parser.OFPActionOutput(port)]
                    self.add_flow(datapath, 10, mpls_match, mpls_actions)
                    time.sleep(0.01)

    def add_flow(self, datapath, priority, match, actions, table_id=0, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    table_id=table_id,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    table_id=table_id,
                                    match=match, instructions=inst)
        self.logger.info("Sending new flow to add: %s", mod)
        datapath.send_msg(mod)

    def mod_flow(self, datapath, priority, match, actions, table_id=0, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    command=ofproto.OFPFC_MODIFY,
                                    table_id=table_id,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, 
                                    command=ofproto.OFPFC_MODIFY,
                                    table_id=table_id,
                                    instructions=inst)
        self.logger.info("Sending new flow to mod: %s", mod)
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
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # ignore lldp packet
            return
        
        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        
        # Figure out environment
        links = api.get_all_link(self)
        switches = api.get_all_switch(self)
        hosts = api.get_all_host(self)
        
        arp_pkt = None
        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            arp_pkt = pkt.get_protocols(arp.arp)[0]
            if arp_pkt.opcode == arp.ARP_REQUEST:
                # Send to ARP proxy. Cannot perform NIx routing until both hosts
                # are known by the controller
                self.ArpProxy (msg.data, datapath, in_port, links, switches, hosts)
                return
            elif arp_pkt.opcode == arp.ARP_REPLY:
                self.ArpReply (msg.data, datapath, arp_pkt.dst_ip, links, switches, hosts)
                return
        
        self.logger.info("%s: packet in %s %s %s %s %s", time.time(), dpid, src, dst, in_port, eth.ethertype)
        
        # Start nix vector code        
        numNodes = len(switches) + len(hosts)
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
        parentVec = {}
        foundIt = self.BFS (numNodes, srcSwitch, dstSwitch,
                                       links, switches, hosts, parentVec)
        
        sdnNix = []
        nixVector = []
        if foundIt:
            self.BuildNixVector (parentVec, srcSwitch, dstSwitch, links, switches, hosts, nixVector, sdnNix)
            
            sdnNix.insert(0, (dstSwitch, dstNode.port.port_no))
            self.sendNixPacket (ofproto, parser, srcSwitch, sdnNix, eth, msg)

            #self.modLastHop (ofproto, parser, dstSwitch, dstNode.port.port_no, eth, msg)
        
    def ArpProxy (self, data, datapath, in_port, links, switches, hosts):
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

    def ArpReply (self, data, datapath, dst_ip, links, switches, hosts):
        for host in hosts:
            for switch in switches:
                # Push an ARP reply out of the appropriate switch port
                if host.port.dpid == switch.dp.id and host.ipv4.count(dst_ip):
                    actions = [switch.dp.ofproto_parser.OFPActionOutput(host.port.port_no)]
                    out = switch.dp.ofproto_parser.OFPPacketOut(datapath=switch.dp,
                                              buffer_id=switch.dp.ofproto.OFP_NO_BUFFER,
                                              in_port=switch.dp.ofproto.OFPP_CONTROLLER,
                                              actions=actions, data=data)

                    self.logger.info("%s: Sending ARP Reply: dpid=%s, port=%s", time.time(), switch.dp.id, host.port.port_no)
                    switch.dp.send_msg(out)
        
    def BFS (self, nNodes, srcSwitch, dstSwitch, links, switches, hosts, parentVector):
        greyNodeList = [ srcSwitch ]
        
        parentVector[srcSwitch.dp.id] = greyNodeList[0]
        while len(greyNodeList) != 0:
            currNode = greyNodeList[0]
            if (currNode == dstSwitch):
                return True
              
            for link in links:
                if link.src.dpid == currNode.dp.id:
                    if not link.dst.is_live():
                        continue
                
                    if parentVector.get(link.dst.dpid) == None:
                        parentVector[link.dst.dpid] = currNode
                        currSwitch = [switch for switch in switches if switch.dp.id == link.dst.dpid][0]
                        greyNodeList.append(currSwitch)
            del(greyNodeList[0])
                         
        return False
    
    def BuildNixVector (self, parentVector, srcSwitch, dstSwitch, links, switches, hosts, nixVector, sdnNix):
        if srcSwitch == dstSwitch:
            return True
        
        if parentVector.get(dstSwitch.dp.id) == None:
            return False
        
        parentSwitch = parentVector[dstSwitch.dp.id]
        destId = 0
        totalNeighbors = len([host for host in hosts if host.port.dpid == parentSwitch.dp.id])
        offset = totalNeighbors
        for link in links:
            if link.src.dpid == parentSwitch.dp.id:
                remoteSwitch = [switch for switch in switches if link.dst.dpid == switch.dp.id][0]
                 
                if remoteSwitch == dstSwitch:
                    sdnNix.append((parentSwitch, link.src.port_no))
                    destId = totalNeighbors + offset
                offset += 1
                totalNeighbors += 1
                
        if totalNeighbors > 1:
            newNix = [int(c) for c in self.bin(destId)[2:]]
            nixVector.extend(newNix)
        self.logger.info("SDN Nix: %s", sdnNix)
        return self.BuildNixVector(parentVector, srcSwitch, parentSwitch, links, switches, hosts, nixVector, sdnNix)
    
    def modLastHop (self, ofproto, parser, switch, port_no, eth, msg):
        mpls_match = parser.OFPMatch()
        mpls_match.append_field(ofproto.OXM_OF_ETH_TYPE, ether_types.ETH_TYPE_MPLS)
        mpls_match.append_field(ofproto.OXM_OF_MPLS_LABEL, port_no)
        mpls_actions = [parser.OFPActionPopMpls(),
                        parser.OFPActionOutput(port_no)]
        
        self.mod_flow(switch.dp, 10, mpls_match, mpls_actions)                    
        self.logger.info("%s: Sending Nix rule: dpid=%s, port=%s", time.time(), switch.dp.id, port_no)

    def sendNixPacket(self, ofproto, parser, srcSwitch, sdnNix, eth, msg):
        # Only set up rule to change eth_type if this will not be a single hop
        ps_actions = [parser.OFPActionPushMpls()]
        if len(sdnNix) > 0:
            ps_inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, ps_actions),
                       #parser.OFPInstructionWriteMetadata(mac.haddr_to_int(eth.dst), UINT64_MAX),
                       parser.OFPInstructionGotoTable(table_id=1)]
            ps_match = parser.OFPMatch(eth_src=eth.src,
                                       eth_dst=eth.dst)
            ps_mod = parser.OFPFlowMod(datapath=srcSwitch.dp, priority=20,
                                       match=ps_match, instructions=ps_inst)
            srcSwitch.dp.send_msg(ps_mod)
            
        actions = []
        out_port = 0
        first = 1        
        for curNix in sdnNix:
            if curNix[0] == srcSwitch:
                # Save the output port from the source switch
                out_port = curNix[1]
            else:
                self.logger.info ("Switch %s send out port %s", curNix[0].dp.id, curNix[1])
                # Only set fields since we added the MPLS header on the first instruction
                if first != 1:
                    actions.append(parser.OFPActionPushMpls())
                actions.append(parser.OFPActionSetField(mpls_ttl=64))
                actions.append(parser.OFPActionSetField(mpls_label=curNix[1]))
                first = 0
        actions.append(parser.OFPActionOutput(out_port))

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        match = parser.OFPMatch()
        match.append_field(ofproto.OXM_OF_ETH_TYPE, ether_types.ETH_TYPE_MPLS)
        match.append_field(ofproto.OXM_OF_ETH_SRC, eth.src)
        match.append_field(ofproto.OXM_OF_ETH_DST, eth.dst)
        #match.append_field(ofproto.OXM_OF_METADATA, mac.haddr_to_int(eth.dst), UINT64_MAX)
        mod = parser.OFPFlowMod(datapath=srcSwitch.dp, priority=10, table_id=1,
                                match=match, instructions=inst)
        self.logger.info("Sending new flow to add: %s", mod)
        srcSwitch.dp.send_msg(mod)
        
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        if len(sdnNix) > 1:
            actions.insert(0, ps_actions[0])
        out = parser.OFPPacketOut(datapath=srcSwitch.dp, buffer_id=msg.buffer_id,
                                  in_port=msg.match['in_port'],
                                  actions=actions, data=data)
        srcSwitch.dp.send_msg(out)
        
    def bin(self, s):
        return str(s) if s<=1 else bin(s>>1) + str(s&1)
