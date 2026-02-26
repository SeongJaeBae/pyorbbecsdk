#!/bin/bash

echo "[1] Kill video users"
sudo fuser -k /dev/video* 2>/dev/null

echo "[2] USB power cycle"
echo '2-3.2' | sudo tee /sys/bus/usb/drivers/usb/unbind
sleep 1
echo '2-3.2' | sudo tee /sys/bus/usb/drivers/usb/bind

echo "[DONE] Orbbec reset complete"