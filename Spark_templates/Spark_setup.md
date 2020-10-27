# mini-Spark computer cluster setup tutorial

## Section 1. Initial system configuration and network settings

System images:

* [CentOS-7-x86_64-DVD-1908](http://isoredirect.centos.org/centos/7/isos/x86_64/) (for master node - qnode0, and worker node - qnode1)

* [Raspberry Pi3 64-bit OS](https://github.com/bamarni/pi64/releases) (for worker nodes - qnode2 and qnode3)

qnode0 and qnode1 configurations:
```
PARTITION       SPACE       FILE_SYSTEM
/               256G        ext4
/work           128G        ext4
/home           64G         ext4
swap            16G         swap

qnode0:     1 sockets x 8 Cores x 2 threads = 16 CPUs
            64G RAM

qnode1:     2 socket  x 6 Cores x 2 threads = 24 CPUs
            24G RAM
```

qnode2 and qnode3 configurations:
```
PARTITION       SPACE       FILE_SYSTEM
/               64G         ext4
/work           200G        nfs4

1 socket x 4 CPUs x 1 thread = 4 CPUs
1G RAM
```

Update software repository

```
yum update (CentOS)
or 
apt update (Raspbian)
```

Enable ssh on RasPi 64-bit machines

```
apt install openssh-server
```

For qnode1-3 that will be worker nodes later (to save memory):
```
systemctl set-default multi-user.target
```

Change the default user name (**pi**) on RasPi, you may first login as root

```
usermod -l rwang -d /home/rwang -m pi
groupmod -n rwang pi
```

Check whether user **pi** has been replaced by **rwang** in these following files
```
/etc/group
/etc/gshadow
/etc/passwd
/etc/shadow
```

After the previous steps, you should have the same system admin user on all three machines, i.e.

```
id rwang
>> uid=1000(rwang) gid=1000(rwang) groups=1000(rwang),10(wheel)
```

Next, we change the hostname for the nodes, which is located in `/etc/hostname `.

We set the hostnames to be `qnode0`, `qnode1`, `qnode2`, and `qnode3`

Then edit `/etc/hosts` to make them look like
```
[rwang@qnode0 ~]$ cat /etc/hosts
>> 127.0.0.1       localhost localhost.localdomain localhost4 localhost4.localdomain4
   ::1             localhost localhost.localdomain localhost6 localhost6.localdomain6

   127.0.1.1       qnode0
   129.105.37.62   qnode0  qnode0.local  qnode0.lan
   129.105.37.153  qnode1  qnode1.local  qnode1.lan
   129.105.37.81   qnode2  qnode2.local  qnode2.lan
   129.105.37.49   qnode3  qnode3.local  qnode3.lan

=================================================================================

[rwang@qnode1 ~]$ cat /etc/hosts
>> 127.0.0.1       localhost localhost.localdomain localhost4 localhost4.localdomain4
   ::1             localhost localhost.localdomain localhost6 localhost6.localdomain6

   127.0.1.1       qnode1
   129.105.37.62   qnode0  qnode0.local  qnode0.lan
   129.105.37.153  qnode1  qnode1.local  qnode1.lan
   129.105.37.81   qnode2  qnode2.local  qnode2.lan
   129.105.37.49   qnode3  qnode3.local  qnode3.lan

=================================================================================

rwang@qnode2:~$ cat /etc/hosts
>> 127.0.1.1       raspberrypi
   127.0.0.1	   localhost
   ::1		       localhost ip6-localhost ip6-loopback
   ff02::1		   ip6-allnodes
   ff02::2		   ip6-allrouters

   127.0.1.1       qnode2
   129.105.37.62   qnode0  qnode0.local  qnode0.lan
   129.105.37.153  qnode1  qnode1.local  qnode1.lan
   129.105.37.81   qnode2  qnode2.local  qnode2.lan
   129.105.37.49   qnode3  qnode3.local  qnode3.lan

=================================================================================

rwang@qnode3:~$ cat /etc/hosts
>> 127.0.1.1       raspberrypi
   127.0.0.1	   localhost
   ::1		       localhost ip6-localhost ip6-loopback
   ff02::1		   ip6-allnodes
   ff02::2		   ip6-allrouters

   127.0.1.1       qnode3
   129.105.37.62   qnode0  qnode0.local  qnode0.lan
   129.105.37.153  qnode1  qnode1.local  qnode1.lan
   129.105.37.81   qnode2  qnode2.local  qnode2.lan
   129.105.37.49   qnode3  qnode3.local  qnode3.lan

```


Now we setup ssh between nodes. For each node, do
```
ssh-keygen -t rsa
```
Then copy the content in `~/.ssh/id_rsa.pub`(from the node that you are currently working on) into the `~/.ssh/authorized_keys` of the other three nodes.

Firewall should be disabled on CentOS (not installed by default on Rasbian)
```
systemctl stop firewalld
systemctl disable firewalld
```


Setup network file system (NFS) on the master node (qnode0)
```
yum -y install nfs-utils
```

Setup NFS on slave nodes
```
yum -y install nfs-utils (CentOS)
apt install nfs-common (Raspbian)
systemctl status nfs-kernel-server (Raspbian)
```


Export work directory through NFS, add the following lines in qnode0 `/etc/exports`
```
/work qnode1(rw,sync,no_root_squash,no_subtree_check)
/work qnode2(rw,sync,no_root_squash,no_subtree_check)
/work qnode3(rw,sync,no_root_squash,no_subtree_check)
```

Mount work directory on slave nodes (qnode1-3), add the following line in `/etc/fstab`
```
qnode0:/work /work nfs auto,nofail,noatime,nolock,intr,tcp,actimeo=1800 0 0
```

## Section 2. Install and setup Slurm job scheduler

Create global users for Slurm and Munge
```
groupadd -g 1001 slurm
useradd -m -d /home/slurm -u 1001 -g slurm -s /bin/bash slurm
groupadd -g 1002 munge
useradd -m -d /var/lib/munge -u 1002 -g munge -s /sbin/nologin munge
```

Install munge 
```
yum install epel-release (CentOS)
yum install munge munge-libs munge-devel (CentOS)
apt install libmunge-dev libmunge2 munge (Raspbian)

chown -R rwang:munge /etc/munge
dd if=/dev/urandom bs=1 count=1024 > /etc/munge/munge.key
chown munge: /etc/munge/munge.key
chmod 400 /etc/munge/munge.key
scp /etc/munge/munge.key root@qnode1:/etc/munge
scp /etc/munge/munge.key root@qnode2:/etc/munge
scp /etc/munge/munge.key root@qnode3:/etc/munge
chown -R munge: /etc/munge/ /var/log/munge/
chmod 0700 /etc/munge/ /var/log/munge/
systemctl enable munge
systemctl start munge
```

Run checks (on qnode0)
```
munge -n | unmunge
munge -n | ssh qnode1 unmunge
munge -n | ssh qnode2 unmunge
munge -n | ssh qnode3 unmunge
```

Sample output when munge is installed successfully
```
STATUS:           Success (0)
ENCODE_HOST:      raspberrypi (127.0.1.1)
ENCODE_TIME:      2019-07-23 17:43:45 +0000 (1563903825)
DECODE_TIME:      2019-07-23 17:43:41 +0000 (1563903821)
TTL:              300
CIPHER:           aes128 (4)
MAC:              sha1 (3)
ZIP:              none (0)
UID:              rwang (1000)
GID:              rwang (1000)
LENGTH:           0
```

Install GCC compiler
```
yum install gcc (CentOS)
yum install gcc-c++ (CentOS)
apt install gcc (Raspbian)
```

Install Mariadb on qnode0 (XDMod requires Mariadb instead of MySQL)
```
yum install mariadb-server mariadb-devel
systemctl start mariadb
systemctl enable mariadb
mysql_secure_installation
MariaDB [(none)]> CREATE USER 'slurm'@'localhost' IDENTIFIED BY 'MYPASSWORD';
MariaDB [(none)]> GRANT ALL ON *.* TO 'slurm'@'localhost';
```

Install deps on CentOS
```
yum install pam-devel
yum install hdf5-devel
yum install glib*
yum install gtk+-devel gtk2-devel
yum install lua-devel
yum install libcurl-devel
yum install json-parser-devel
```

Install deps on Raspbian
```
apt install libpam0g-dev
apt install libhdf5-serial-dev
apt install glibc-*
apt install gtk2.0
apt install libssl-dev
apt install liblua5.1-0-dev liblua50-dev liblualib50-dev
apt install libcurl4-openssl-dev
```

Install Slurm
```
su slurm

# download, configure, compile, and install
wget https://download.schedmd.com/slurm/slurm-18.08.7.tar.bz2
tar -jxvf slurm-18.08.7.tar.bz2
cd slurm-18.08.7/
./configure --prefix=/opt/slurm --sysconfdir=/etc/slurm (note: check the configuration log file to ensure deps are met)
make -j
make install

# post installation
ldconfig -n /opt/slurm/lib/
cp ${SLURM_INSTALLATION_DIR}/etc/slurmctld.service  /etc/systemd/system/ (for head node)
cp ${SLURM_INSTALLATION_DIR}/etc/slurmdbd.service   /etc/systemd/system/ (for storage node, if different from head node)
cp ${SLURM_INSTALLATION_DIR}/etc/slurmd.service  /etc/systemd/system/  (for compute node)
```

Add slurm commands to path
```
secure_path="$secure_path:${SLURM_PATH}/sbin"
```

Enable slurm
```
systemctl enable slurmctld (qnode0)
systemctl enable slurmdbd (qnode0)
systemctl enable slurmd (qnode1-3)
```

Change pid file path in "/etc/systemd/system/multi-user.target.wants/slurm*d.service" (optional)

Add new users
```
useradd -d /home/pi1 -m -s /bin/bash -u ${USERID} ${USERNAME}
groupadd -g ${GROUPID} ${GROUPNAME}
usermod -aG ${GID} ${UID}
```

Manage Slurm cluster name, account, and users
```
sacctmgr add cluster ${CLUSTERNAME}
sacctmgr add account ${ACCTNAME} Description="descriptions.."
sacctmgr add user ${USERNAME} Account=${ACCTNAME}
```

Install XDMod after downloading the rpm files form [here](https://github.com/ubccr/xdmod/releases).
```
yum install epel-release
yum install httpd php php-cli php-mysql php-gd php-mcrypt \
              gmp-devel php-gmp php-pdo php-xml php-pear-Log \
              php-pear-MDB2 php-pear-MDB2-Driver-mysql \
              java-1.7.0-openjdk java-1.7.0-openjdk-devel \
              mariadb-server mariadb cronie logrotate \
              ghostscript postfix
yum install xdmod-x.x.x-x.x.el6.noarch.rpm
```

PhantomJs needs to be installed manually [here](http://phantomjs.org/download.html)

Edit /etc/php.ini, uncomment this line as
```
date.timezone = America/New_York
```

Enable httpd
```
yum -y install httpd
systemctl enable httpd
```

Setup XDMod
```
xdmod-setup
```

In order to emove hierarchical levels, remove all these lines in  ${PATH_TO_XDMOD}/roles.json
```
{
    "realm": "Jobs",
    "group_by": "nsfdirectorate"
},
{
    "realm": "Jobs",
    "group_by": "parentscience"
},
{
    "realm": "Jobs",
    "group_by": "fieldofscience"
},
```

Then update the configuration (this executatble should be in the same bin as other XDMod commands)
```
acl-config 
```

Ingest data from Slurm scheduler
```
xdmod-slurm-helper -r ${CLUSTER_NAME}
xdmod-ingestor
```

Open your browser, go to (you may need to restart httpd service)
```
http://localhost:8080
```


# Spark cluster

```
wget http://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.rpm (CentOS)
yum install scala-2.11.8.rpm (CentOS)

apt install scale (Raspbian)

wget https://www.apache.org/dyn/closer.lua/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz
tar zxvf spark-2.4.3-bin-hadoop2.7.tgz
mv spark-2.4.3-bin-hadoop2.7 /usr/local/spark
```

On qnode0
```
cd /usr/local/spark/conf
cp spark-env.sh.template spark-env.sh

vi spark-env.sh
export SPARK_MASTER_HOST='<MASTER-IP>'
export JAVA_HOME=<Path_of_JAVA_installation>

vi slaves
    qnode1
    qnode2
    qnode3
```

