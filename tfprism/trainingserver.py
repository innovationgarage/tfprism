#! /usr/bin/env python

import argparse
import sys

import tensorflow as tf
import netifaces
import dns.resolver
import pieshell
import multiprocessing
import click

FLAGS = None

def run_server(spec, job_name, task_index):
  print "Starting server /job:%s/task:%s as %s..." % (job_name, task_index, spec[job_name][task_index])
  
  tf.train.Server(
    tf.train.ClusterSpec(spec),
    job_name=job_name,
    task_index=task_index
  ).join()


def generate_tasks(servers, base_port):
  """Input: {"server1": ncpus, "server2":ncpus...}
     Output: (("server1", port1), ("server1", port2)...("serverN", "portM"))
  """
  for server, ncpus in servers:
    for cpuidx in xrange(0, ncpus):
      yield (server, base_port + cpuidx)

  
def generate_cluster(servers, base_port, n_ps_tasks):
  tasks = ["%s:%s" % (server, port) for server, port in generate_tasks(servers, base_port)]
  ps_tasks = tasks[:n_ps_tasks]
  worker_tasks = tasks[n_ps_tasks:]
  return {'ps': ps_tasks, 'worker': worker_tasks}

def find_local_server_idx(servers):
  local_ips = set([netifaces.ifaddresses(iface)[netifaces.AF_INET][0]['addr']
                   for iface in netifaces.interfaces()
                   if netifaces.AF_INET in netifaces.ifaddresses(iface)])
  local_ips.add("127.0.1.1") # Hack for debian
  
  task_ips = [server[0] for server in servers]
  task_ips = [record.address
              for ip in task_ips
              for record in dns.resolver.query(ip, 'A')]
  local_task_ip = iter(local_ips.intersection(set(task_ips))).next()
  
  return task_ips.index(local_task_ip)

def generate_task_indexes(servers, server_idx, n_ps_tasks):
  base_task_idx = sum(s[1] for s in servers[:server_idx])
  server = servers[server_idx]
  for n in xrange(0, server[1]):
    task_idx = base_task_idx + n
    if task_idx >= n_ps_tasks:
      yield "worker", task_idx - n_ps_tasks
    else:
      yield "ps", task_idx

def servers_to_str(servers):
  return ",".join("%s:%s" % s for s in servers)


def str_to_servers(str):
  return [(name, int(ncpus)) for name, ncpus in (s.split(":") for s in str.split(","))]

def introspect_cluster(servernames):
  return ",".join(pieshell.env.parallel("--no-notice", "--nonall", "--line-buffer", "-S", servernames,
                                        'echo -n "$(hostname):"; cat /proc/cpuinfo  | grep "processor" | wc -l'))

def start_cluster(servernames, base_port, n_ps_tasks):
  servers = introspect_cluster(servernames)
  print pieshell.env.parallel(
    '--no-notice', '--nonall', '--line-buffer', '--tag',
    '-S', servernames,
    'nohup tfprism node run --base_port %s --ps_tasks %s %s < /dev/null > tfprism.log 2>&1 & echo "$!" > /var/run/tfprism.pid; sleep 2' % (
      base_port, n_ps_tasks, servers))

def stop_cluster(servernames):
    print pieshell.env.parallel(
    '--no-notice', '--nonall', '--line-buffer', '--tag',
    '-S', servernames,
    "kill -KILL $(cat /var/run/tfprism.pid)" % servers)

def run_node(servers, base_port, n_ps_tasks):
  servers = str_to_servers(servers)
  cluster_spec = generate_cluster(servers, base_port, n_ps_tasks)
  procs = [multiprocessing.Process(target=run_server, args=(cluster_spec, job_name, task_index))
           for job_name, task_index in generate_task_indexes(servers, find_local_server_idx(servers), n_ps_tasks)]
  for proc in procs:
    proc.daemon = True
    proc.start()
  for proc in procs:
    proc.join()

    
@click.group()
def main():
    pass

@main.group()
def node():
    pass

@node.command()
@click.argument("servers")
@click.option('--base_port', default=5600)
@click.option('--ps_tasks', default=1)
def run(servers, base_port, ps_tasks):
  run_node(servers, base_port, ps_tasks)

@main.group()
def cluster():
    pass

@cluster.command()
@click.argument("servers")
@click.option('--base_port', default=5600)
@click.option('--ps_tasks', default=1)
def start(servers, base_port, ps_tasks):
  start_cluster(servers, base_port, ps_tasks)

@cluster.command()
@click.argument("servers")
def stop(servers):
  stop_cluster(servers)

if __name__ == "__main__":
  main()
  
