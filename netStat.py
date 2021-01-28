import numpy as np
## Prep AfterImage cython package
import pyximport
import sys

if sys.platform == 'linux':
    pyximport.install()
else:
    mingw_setup_args = {'options': {'build_ext': {'compiler': 'mingw32'}}}
    pyximport.install(setup_args=mingw_setup_args)

import AfterImage as af


# import AfterImage_NDSS as af
#
# MIT License
#
# Copyright (c) 2018 Yisroel mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class netStat:
    # Datastructure for efficent network stat queries
    # HostLimit: no more that this many Host identifiers will be tracked
    # HostSimplexLimit: no more that this many outgoing channels from each host will be tracked (purged periodically)
    # Lambdas: a list of 'window sizes' (decay factors) to track for each stream. nan resolved to default [5,3,1,.1,.01]
    def __init__(self, Lambdas=np.nan, HostLimit=255, HostSimplexLimit=1000):
        # Lambdas
        if np.isnan(Lambdas):
            self.Lambdas = [5, 3, 1, .1, .01]
        else:
            self.Lambdas = Lambdas
        # HT Limits
        self.HostLimit = HostLimit
        self.SessionLimit = HostSimplexLimit * self.HostLimit * self.HostLimit  # *2 since each dual creates 2 entries in memory
        self.MAC_HostLimit = self.HostLimit * 10
        # HTs
        self.HT_jit = af.incStatDB(limit=self.HostLimit * self.HostLimit)  # H-H Jitter Stats
        self.HT_MI = af.incStatDB(limit=self.MAC_HostLimit)  # MAC-IP relationships
        self.HT_H = af.incStatDB(limit=self.HostLimit)  # Source Host BW Stats
        self.HT_Hp = af.incStatDB(limit=self.SessionLimit)  # Source Host BW Stats
        self.PP_jit = af.incStatDB(limit=self.HostLimit * self.HostLimit)
        self.ST_jit = af.incStatDB(limit=self.MAC_HostLimit)
        self.SCAN = af.incStatDB(limit=self.MAC_HostLimit)
        self.AR = af.incStatDB(limit=self.MAC_HostLimit)
        self.SYN_jit = af.incStatDB(limit=self.MAC_HostLimit)
        self.SR_jit = af.incStatDB(limit=self.MAC_HostLimit)
        self.SSL_jit = af.incStatDB(limit=self.MAC_HostLimit)

    def findDirection(self, IPtype, srcIP, dstIP, eth_src,
                      eth_dst):  # cpp: this is all given to you in the direction string of the instance (NO NEED FOR THIS FUNCTION)
        if IPtype == 0:  # is IPv4
            lstP = srcIP.rfind('.')
            src_subnet = srcIP[0:lstP:]
            lstP = dstIP.rfind('.')
            dst_subnet = dstIP[0:lstP:]
        elif IPtype == 1:  # is IPv6
            src_subnet = srcIP[0:round(len(srcIP) / 2):]
            dst_subnet = dstIP[0:round(len(dstIP) / 2):]
        else:  # no Network layer, use MACs
            src_subnet = eth_src
            dst_subnet = eth_dst
        return src_subnet, dst_subnet

    def updateGetStats(self, IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, datagramSize, timestamp,
                       istls, syn, ack):

        # MAC.IP: Stats on src MAC-IP relationships
        MIstat = np.zeros((3 * len(self.Lambdas, )))
        for i in range(len(self.Lambdas)):
            MIstat[(i * 3):((i + 1) * 3)] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
                                                                           self.Lambdas[i])
        #
        ARstat = np.zeros((3 * len(self.Lambdas, )))
        for i in range(len(self.Lambdas)):
            ARstat[(i * 3):((i + 1) * 3)] = self.AR.update_get_1D_Stats(dstIP, timestamp, datagramSize, self.Lambdas[i])

        STstat_jit = np.zeros((3 * len(self.Lambdas, )))
        for i in range(len(self.Lambdas)):
            STstat_jit[(i * 3):((i + 1) * 3)] = self.ST_jit.update_get_1D_Stats(dstIP, timestamp, 0, self.Lambdas[i],
                                                                                isTypeDiff=True)

        Scanstat = np.zeros((3 * len(self.Lambdas, )))
        for i in range(len(self.Lambdas)):
            Scanstat[(i * 3):((i + 1) * 3)] = self.SCAN.update_get_1D_Stats(srcIP + dstProtocol, 0, self.Lambdas[i],
                                                                            isTypeDiff=True)

        # Host-Host Jitter:
        HHstat_jit = np.zeros((3 * len(self.Lambdas, )))
        for i in range(len(self.Lambdas)):
            HHstat_jit[(i * 3):((i + 1) * 3)] = self.HT_jit.update_get_1D_Stats(srcIP + dstIP, timestamp, 0,
                                                                                self.Lambdas[i], isTypeDiff=True)

        PPstat_jit = np.zeros((3 * len(self.Lambdas, )))
        if srcProtocol == 'arp' or srcProtocol == 'icmp':
            for i in range(len(self.Lambdas)):
                PPstat_jit[(i * 3):((i + 1) * 3)] = self.PP_jit.update_get_1D_Stats(srcIP, timestamp, 0,
                                                                                    self.Lambdas[i],
                                                                                    isTypeDiff=True)
        else:
            for i in range(len(self.Lambdas)):
                PPstat_jit[(i * 3):((i + 1) * 3)] = self.PP_jit.update_get_1D_Stats(dstIP + dstProtocol, timestamp, 0,
                                                                                    self.Lambdas[i], isTypeDiff=True)

        # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        HpHpstat = np.zeros((7 * len(self.Lambdas, )))
        if srcProtocol == 'arp' or srcProtocol == 'icmp':
            for i in range(len(self.Lambdas)):
                HpHpstat[(i * 7):((i + 1) * 7)] = self.HT_Hp.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp,
                                                                                   datagramSize, self.Lambdas[i])
        else:  # some other protocol (e.g. TCP/UDP)
            for i in range(len(self.Lambdas)):
                HpHpstat[(i * 7):((i + 1) * 7)] = self.HT_Hp.update_get_1D2D_Stats(srcIP + srcProtocol,
                                                                                   dstIP + dstProtocol, timestamp,
                                                                                   datagramSize, self.Lambdas[i])
        SRstat = np.zeros((3 * len(self.Lambdas, )))
        for i in range(len(self.Lambdas)):
            SRstat[(i * 3):((i + 1) * 3)] = self.SR_jit.update_get_1D_Stats(srcIP, timestamp, 0, self.Lambdas[i],
                                                                            isTypeDiff=True)

        SSLjit = np.zeros((3 * len(self.Lambdas, )))
        if (istls == 1):
            for i in range(len(self.Lambdas)):
                SSLjit[(i * 3):((i + 1) * 3)] = self.SSL_jit.update_get_1D_Stats(dstIP + dstProtocol, timestamp, 0,
                                                                                 self.Lambdas[i], isTypeDiff=True)

        syn_jit = np.zeros((3 * len(self.Lambdas, )))
        if (syn and ack):
            for i in range(len(self.Lambdas)):
                syn_jit[(i * 3):((i + 1) * 3)] = self.SYN_jit.update_get_1D_Stats(dstIP, timestamp, 0, self.Lambdas[i],
                                                                                  isTypeDiff=True)

        Sjit = np.zeros((3 * len(self.Lambdas, )))
        for i in range(len(self.Lambdas)):
            Sjit[(i * 3):((i + 1) * 3)] = self.PP_jit.update_get_1D_Stats(srcIP, timestamp, 0, self.Lambdas[i],
                                                                          isTypeDiff=True)

        # Host BW: Stats on the srcIP's general Sender Statistics
        Hstat = np.zeros((3 * len(self.Lambdas, )))
        for i in range(len(self.Lambdas)):
            Hstat[(i * 3):((i + 1) * 3)] = self.HT_H.update_get_1D_Stats(srcIP, timestamp, datagramSize,
                                                                         self.Lambdas[i])

        # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        HHstat = np.zeros((7 * len(self.Lambdas, )))
        for i in range(len(self.Lambdas)):
            HHstat[(i * 7):((i + 1) * 7)] = self.HT_H.update_get_1D2D_Stats(srcIP, dstIP, timestamp, datagramSize,
                                                                            self.Lambdas[i])

        return np.concatenate((MIstat, ARstat, STstat_jit, Scanstat, HHstat_jit, PPstat_jit, HpHpstat, SRstat, SSLjit,
                               syn_jit, Sjit, Hstat, HHstat))
        # return np.concatenate((HHstat,HHstat_jit,SRstat,HpHpstat))#100

        # return np.concatenate((MIstat,HHstat_jit,PPstat_jit,ARstat,Scanstat,STstat_jit))
        # return np.concatenate((MIstat,HHstat_jit,PPstat_jit,ARstat,Scanstat,STstat_jit))  # concatenation of stats into one stat vector
        # return ARstat
        # return np.concatenate((MIstat, HpHpstat,HHstat_jit,ARstat,STstat_jit,syn_jit))  # concatenation of stats into one stat vector
        # return np.concatenate((HHstat, MIstat, HHstat_jit,HpHpstat))
        # return np.concatenate((MIstat,HHstat,HHstat_jit,PPstat_jit,ARstat,STstat_jit))
        # return np.concatenate((MIstat,HHstat,HHstat_jit,STstat_jit,Scanstat,ARstat))
        # return HpHpstat

    def getNetStatHeaders(self):
        MIstat_headers = []
        Hstat_headers = []
        HHstat_headers = []
        HHjitstat_headers = []
        HpHpstat_headers = []
        PPjitstat_headers = []
        STjit_headers = []
        Scan_headers = []
        AR_headers = []
        SYN_headers = []

        # for i in range(len(self.Lambdas)):
        # Hstat_headers += ["HS_"+h for h in self.HT_H.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
        # MIstat_headers += ["MI_dir_"+h for h in self.HT_MI.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
        # HHstat_headers += ["HH_"+h for h in self.HT_H.getHeaders_1D2D(Lambda=self.Lambdas[i],IDs=None,ver=2)]
        # HHjitstat_headers += ["HH_jit_"+h for h in self.HT_jit.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
        # PPjitstat_headers += ["PP_jit_" + h for h in self.PP_jit.getHeaders_1D(Lambda=self.Lambdas[i], ID=None)]
        # HpHpstat_headers += ["HpHp_" + h for h in self.HT_Hp.getHeaders_1D2D(Lambda=self.Lambdas[i], IDs=None, ver=2)]
        # STjit_headers += ["STjit_"+ h for h in self.ST_jit.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
        # Scan_headers += ["Scan_"+ h for h in self.SCAN.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
        # AR_headers += ["AR_"+ h for h in self.AR.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
        # SYN_headers += ["SYN_" + h for h in self.SYN_jit.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
        # print(MIstat_headers)
        # print(len(MIstat_headers))
        # return MIstat_headers + Hstat_headers + HHstat_headers + HHjitstat_headers + HpHpstat_headers+PPjitstat_headers+STjit_headers+Scan_headers+AR_headers+SYN_headers
