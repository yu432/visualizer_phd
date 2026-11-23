"""Self-test for mem_guard.py: allocate incompressible memory until the guard kills it.

Random bytes are used on purpose. A run filled with a repeated byte is compressed by macOS,
so its resident size stays far below what it really occupies and the guard would look broken.
"""
import os
import signal
import sys
import time

if len(sys.argv) > 2 and sys.argv[2] == "ignore":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # exercise the SIGKILL fallback

chunks = []
target = int(float(sys.argv[1]) * 1024 ** 3)
step = 128 * 1024 * 1024
done = 0
while done < target:
    block = bytearray(os.urandom(step))
    chunks.append(block)
    done += len(block)
    print("hog: %.2f GiB" % (done / 1024 ** 3), flush=True)
    time.sleep(0.1)
print("hog: finished without being killed", flush=True)
time.sleep(30)
