# Detecção de Intrusão - CIC IoT Dataset 2023


## Dataset
O dataset contém 33 ataques executados em uma topologia IoT composta por 105 dispositivos.

### Download
```bash
wget http://205.174.165.80/IOTDataset/CIC_IOT_Dataset2023/CICIOT/csv/CICIoT2023.zip && unzip CICIoT2023.zip -d CICIoT2023
```

### Features 
| Feature         | Description                                                                                                            |
| --------------- | ---------------------------------------------------------------------------------------------------------------------- |
| ts              | Timestamp of first packet in flow                                                                                      |
| flow_duration   | Time between first and last packet received in flow                                                                    |
| Header_Length   | Length of packet header in bits                                                                                        |
| Protocol        | Type 	Protocol numbers, as defined by the IANA. Ex: 1 = ICMP, 6 = TCP                                                |
| Duration        | Time-to-Live (ttl)                                                                                                     |
| Rate            | Rate of packet transmission in a flow                                                                                  |
| Srate           | Rate of outbound (sent) packets transmission in a flow                                                                 |
| Drate           | Rate of inbound (received) packets transmission in a flow                                                              |
| fin_flag_number | Fin flag value                                                                                                         |
| syn_flag_number | Syn flag value                                                                                                         |
| rst_flag_number | Rst flag value                                                                                                         |
| psh_flag_numbe  | Psh flag value                                                                                                         |
| ack_flag_number | Ack flag value                                                                                                         |
| ece_flag_number | Ece flag value                                                                                                         |
| cwr_flag_number | Cwr flag value                                                                                                         |
| ack_count       | Number of packets with ack flag set in the same flow                                                                   |
| syn_count       | Number of packets with syn flag set in the same flow                                                                   |
| fin_count       | Number of packets with fin flag set in the same flow                                                                   |
| urg_count       | Number of packets with urg flag set in the same flow                                                                   |
| rst_count       | Number of packets with rst flag set in the same flow                                                                   |
| HTTP            | Indicates if the application layer protocol is HTTP                                                                    |
| HTTPS           | Indicates if the application layer protocol is HTTPS                                                                   |
| DNS             | Indicates if the application layer protocol is DNS                                                                     |
| Telnet          | Indicates if the application layer protocol is Telnet                                                                  |
| SMTP            | Indicates if the application layer protocol is SMTP                                                                    |
| SSH             | Indicates if the application layer protocol is SSH                                                                     |
| IRC             | Indicates if the application layer protocol is IRC                                                                     |
| TCP             | Indicates if the transport layer protocol is TCP                                                                       |
| UDP             | Indicates if the transport layer protocol is UDP                                                                       |
| DHCP            | Indicates if the application layer protocol is DHCP                                                                    |
| ARP             | Indicates if the link layer protocol is ARP                                                                            |
| ICMP            | Indicates if the network layer protocol is ICMP                                                                        |
| IPv             | Indicates if the network layer protocol is IP                                                                          |
| LLC             | Indicates if the link layer protocol is LLC                                                                            |
| Tot_sum         | Summation of packets lengths in flow                                                                                   |
| Min             | Minimum packet length in the flow                                                                                      |
| Max             | Maximumpacket length in the flow                                                                                       |
| AVG             | Average packet length in the flow                                                                                      |
| Std             | Standard deviation of packet length in the flow                                                                        |
| Tot_size        | Packet’s length                                                                                                        |
| IAT             | The time difference with the previous packet                                                                           |
| Number          | The number of packets in the flow                                                                                      |
| Magnitude       | sqrt(Average of the lengths of incoming packets in the flow + average of the lengths of outgoing packets in the flow)  |
| Radius          | sqrt(Variance of the lengths of incoming packets in the flow +variance of the lengths of outgoing packets in the flow) |
| Covariance      | Covariance of the lengths of incoming and outgoing packets                                                             |
| Variance        | Variance of the lengths of incoming packets in the flow/variance of the lengths of outgoing packets in the flow        |
| Weight          | Number of incoming packets × Number of outgoing packets                                                                |
| label           | Notes the type of attack being run or 'BenignTraffic' for no attack run                                                |

<p align="center">
// Your content
</p>
