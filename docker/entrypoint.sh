#!/bin/bash

IP_ADDR="$(ip -o -4 addr list eth0 | perl -n -e 'if (m{inet\s([\d\.]+)\/\d+\s}xms) { print $1 }')"
echo "Container IP Address: $IP_ADDR"
#export MASTER="spark://${IP_ADDR}:7077"
export SPARK_LOCAL_IP="${IP_ADDR}"
export SPARK_PUBLIC_DNS="${IP_ADDR}"

umount /etc/hosts

exec ipython -i $@
