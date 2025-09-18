#!/usr/bin/env python3
"""
srsRAN ZMQ Connection Tester
This script helps analyze and test ZMQ connections used by srsRAN for gNodeB-UE communication.
Based on the logs, it appears srsRAN uses ZMQ for RF frontend simulation.
"""

import zmq
import numpy as np
import time
import threading
import argparse
import struct
import sys

class SrsRanZmqTester:
    def __init__(self):
        self.context = zmq.Context()
        self.running = False
        
    def test_zmq_discovery(self):
        """Try to discover ZMQ endpoints by testing common patterns"""
        print("=== ZMQ Endpoint Discovery ===")
        
        # Common srsRAN ZMQ ports and patterns
        test_endpoints = [
            "tcp://127.0.0.1:2000"
        ]
        
        active_endpoints = []
        
        for endpoint in test_endpoints:
            try:
                # Test as REP (server) socket
                socket = self.context.socket(zmq.REP)
                socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
                socket.bind(endpoint)
                print(f"✓ Can bind to {endpoint} (REP)")
                socket.close()
                active_endpoints.append((endpoint, "REP", "Available"))
            except zmq.ZMQError as e:
                if "Address already in use" in str(e):
                    print(f"⚠ {endpoint} already in use (likely srsRAN)")
                    active_endpoints.append((endpoint, "Unknown", "In Use"))
                else:
                    print(f"✗ {endpoint} - {e}")
        
        return active_endpoints
    
    def simulate_rf_frontend(self, tx_endpoint, rx_endpoint, duration=10):
        """Simulate RF frontend that communicates with srsRAN"""
        print(f"\n=== Simulating RF Frontend ===")
        print(f"TX Endpoint: {tx_endpoint}")
        print(f"RX Endpoint: {rx_endpoint}")
        
        self.running = True
        
        # Start RX thread (provides samples to srsRAN)
        rx_thread = threading.Thread(target=self._rx_handler, args=(rx_endpoint,))
        rx_thread.daemon = True
        rx_thread.start()
        
        # Start TX thread (receives samples from srsRAN)  
        tx_thread = threading.Thread(target=self._tx_handler, args=(tx_endpoint,))
        tx_thread.daemon = True
        tx_thread.start()
        
        print(f"Running for {duration} seconds...")
        time.sleep(duration)
        self.running = False
        
        print("Stopping simulation...")
        time.sleep(1)
    
    def _rx_handler(self, endpoint):
        """Handle RX - send samples TO srsRAN"""
        try:
            socket = self.context.socket(zmq.PUB)  # Publisher
            socket.bind(endpoint)
            time.sleep(0.1)  # Let socket settle
            
            sample_rate = 23.04e6  # Common LTE sample rate
            samples_per_packet = 11520  # From your logs
            
            print(f"RX: Publishing samples on {endpoint}")
            
            packet_count = 0
            while self.running:
                # Generate dummy IQ samples (complex64)
                samples = np.random.normal(0, 0.1, samples_per_packet) + \
                         1j * np.random.normal(0, 0.1, samples_per_packet)
                
                # Convert to bytes (complex64 = 2 float32)
                samples_bytes = samples.astype(np.complex64).tobytes()
                
                # Send samples
                socket.send(samples_bytes, zmq.NOBLOCK)
                packet_count += 1
                
                if packet_count % 1000 == 0:
                    print(f"RX: Sent {packet_count} packets")
                
                time.sleep(0.001)  # 1ms between packets
                
        except Exception as e:
            print(f"RX Handler error: {e}")
        finally:
            socket.close()
    
    def _tx_handler(self, endpoint):
        """Handle TX - receive samples FROM srsRAN"""
        try:
            socket = self.context.socket(zmq.SUB)  # Subscriber
            socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all
            socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout
            socket.connect(endpoint)
            
            print(f"TX: Listening for samples on {endpoint}")
            
            packet_count = 0
            while self.running:
                try:
                    data = socket.recv()
                    packet_count += 1
                    
                    if packet_count % 1000 == 0:
                        print(f"TX: Received {packet_count} packets ({len(data)} bytes)")
                        
                except zmq.Again:
                    continue  # Timeout, keep trying
                    
        except Exception as e:
            print(f"TX Handler error: {e}")
        finally:
            socket.close()
    
    def sniff_zmq_traffic(self, endpoints, duration=10):
        """Sniff ZMQ traffic on specified endpoints"""
        print(f"\n=== ZMQ Traffic Sniffer ===")
        
        for endpoint in endpoints:
            thread = threading.Thread(target=self._sniff_endpoint, args=(endpoint, duration))
            thread.daemon = True
            thread.start()
        
        time.sleep(duration)
    
    def _sniff_endpoint(self, endpoint, duration):
        """Sniff traffic on a specific endpoint"""
        try:
            # Try different socket types
            socket_types = [
                (zmq.SUB, "SUB"),
                (zmq.PULL, "PULL"),
                (zmq.REQ, "REQ")
            ]
            
            for sock_type, name in socket_types:
                try:
                    socket = self.context.socket(sock_type)
                    socket.setsockopt(zmq.RCVTIMEO, 1000)
                    
                    if sock_type == zmq.SUB:
                        socket.setsockopt(zmq.SUBSCRIBE, b"")
                    
                    socket.connect(endpoint)
                    
                    start_time = time.time()
                    packet_count = 0
                    
                    while time.time() - start_time < duration:
                        try:
                            data = socket.recv()
                            packet_count += 1
                            
                            if packet_count == 1:
                                print(f"📡 {endpoint} ({name}): First packet {len(data)} bytes")
                            elif packet_count % 100 == 0:
                                print(f"📡 {endpoint} ({name}): {packet_count} packets")
                                
                        except zmq.Again:
                            continue
                    
                    if packet_count > 0:
                        print(f"✓ {endpoint} ({name}): Total {packet_count} packets")
                    
                    socket.close()
                    break  # Found working socket type
                    
                except zmq.ZMQError:
                    socket.close()
                    continue
                    
        except Exception as e:
            print(f"Sniff error on {endpoint}: {e}")
    
    def analyze_sample_format(self, data):
        """Analyze the format of received samples"""
        if len(data) < 8:
            return "Too short to analyze"
        
        # Try different interpretations
        results = []
        
        # Complex64 (8 bytes per sample)
        if len(data) % 8 == 0:
            samples = len(data) // 8
            results.append(f"Complex64: {samples} samples")
        
        # Complex32 (4 bytes per sample) 
        if len(data) % 4 == 0:
            samples = len(data) // 4
            results.append(f"Complex32: {samples} samples")
        
        # Float32 (4 bytes per value, 2 per complex sample)
        if len(data) % 8 == 0:
            samples = len(data) // 8
            results.append(f"Float32 IQ pairs: {samples} samples")
        
        return " | ".join(results)

def main():
    parser = argparse.ArgumentParser(description="srsRAN ZMQ Connection Tester")
    parser.add_argument("--discover", action="store_true", 
                       help="Discover active ZMQ endpoints")
    parser.add_argument("--simulate", action="store_true",
                       help="Simulate RF frontend")
    parser.add_argument("--sniff", action="store_true",
                       help="Sniff ZMQ traffic")
    parser.add_argument("--tx-endpoint", default="tcp://localhost:2000",
                       help="TX endpoint (default: tcp://localhost:2000)")
    parser.add_argument("--rx-endpoint", default="tcp://localhost:2001", 
                       help="RX endpoint (default: tcp://localhost:2001)")
    parser.add_argument("--duration", type=int, default=10,
                       help="Test duration in seconds (default: 10)")
    parser.add_argument("--save-samples", action="store_true", default=True,
                       help="Save captured samples to files (default: True)")
    parser.add_argument("--no-save", dest='save_samples', action="store_false",
                       help="Don't save captured samples to files")
    
    args = parser.parse_args()
    
    tester = SrsRanZmqTester()
    
    try:
        if args.discover:
            endpoints = tester.test_zmq_discovery()
            print(f"\nFound {len(endpoints)} endpoints")
            
        if args.sniff:
            endpoints = [args.tx_endpoint, args.rx_endpoint]
            tester.sniff_zmq_traffic(endpoints, args.duration)
            
        if args.simulate:
            tester.simulate_rf_frontend(args.tx_endpoint, args.rx_endpoint, args.duration)
            
        if not any([args.discover, args.sniff, args.simulate]):
            print("No action specified. Use --help for options.")
            print("\nQuick start:")
            print("python3 srsran_zmq_tester.py --discover")
            print("python3 srsran_zmq_tester.py --sniff --duration 5")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        tester.context.term()

if __name__ == "__main__":
    main()
