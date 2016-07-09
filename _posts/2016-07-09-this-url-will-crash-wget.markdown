---
layout: post
title:  "This URL will crash your Ubuntu wget"
date:   2016-07-09 11:00:00 +0800
categories: security linux
---

Today something funny happened to me, I was crawling the web for PDFs files when I noticed my Ubuntu 16.04 wget command segfaulting, which is a pretty rare thing considering how well those commands are tested against input.

On Ubuntu 16.04 a wget version 1.17.1 is shipped, 

`GNU Wget 1.17.1 built on linux-gnu.`

so I grabbed the sources from wget website to debug / investigate, or give a quick run on it with ASAN on, because I'm lazy...

On the wget website there is the version 1.18 which does not exibit this problem. So I assume or it was fixed or the code changed.

Anyway just sharing this because I think it's funny and rare to see!

The culprit URL is : `http://ia600208.us.archive.org/23/items/sarabia_20160316_0705/%d9%84%db%95%20%d8%aa%db%86%d9%be%d8%ae%d8%a7%d9%86%db%95%d9%88%db%95%20%d8%a8%db%86%20%d8%b9%db%95%d8%b1%d8%b9%db%95%d8%b1.pdf`

which is a very peculiar filename, with arabics characters. In fact I didn't even open it or can read the title, if anyone knows what the title means let me know, I was just downloading a bunch of stuff from `archive.org`.

# ASAN trace

{% highlight text %}
➜  wget-1.17.1 ./src/wget http://ia600208.us.archive.org/23/items/sarabia_20160316_0705/%d9%84%db%95%20%d8%aa%db%86%d9%be%d8%ae%d8%a7%d9%86%db%95%d9%88%db%95%20%d8%a8%db%86%20%d8%b9%db%95%d8%b1%d8%b9%db%95%d8%b1.pdf
--2016-07-09 23:38:04--  http://ia600208.us.archive.org/23/items/sarabia_20160316_0705/%d9%84%db%95%20%d8%aa%db%86%d9%be%d8%ae%d8%a7%d9%86%db%95%d9%88%db%95%20%d8%a8%db%86%20%d8%b9%db%95%d8%b1%d8%b9%db%95%d8%b1.pdf
Resolving ia600208.us.archive.org (ia600208.us.archive.org)... 207.241.227.228
Connecting to ia600208.us.archive.org (ia600208.us.archive.org)|207.241.227.228|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 444923 (434K) [application/pdf]
Saving to: ‘\331%84\333%95 ت\333%86پخا\331%86\333%95\331%88\333%95 ب\333%86 ع\333%95رع\333%95ر.pdf.2’

=================================================================
==42306==ERROR: AddressSanitizer: negative-size-param: (size=-4)
    #0 0x7fa89f7c8e72  (/usr/lib/x86_64-linux-gnu/libasan.so.3+0x47e72)
    #1 0x4c986f in memset /usr/include/x86_64-linux-gnu/bits/string3.h:90
    #2 0x4c986f in create_image /media/bob/e4109b52-3574-43a8-b95d-33b3494128de/misc/wget-1.17.1/src/progress.c:1167
    #3 0x4cbdb6 in bar_create /media/bob/e4109b52-3574-43a8-b95d-33b3494128de/misc/wget-1.17.1/src/progress.c:602
    #4 0x4dd0ae in fd_read_body /media/bob/e4109b52-3574-43a8-b95d-33b3494128de/misc/wget-1.17.1/src/retr.c:274
    #5 0x4826bc in read_response_body /media/bob/e4109b52-3574-43a8-b95d-33b3494128de/misc/wget-1.17.1/src/http.c:1682
    #6 0x49be1d in gethttp /media/bob/e4109b52-3574-43a8-b95d-33b3494128de/misc/wget-1.17.1/src/http.c:3753
    #7 0x4a1aaf in http_loop /media/bob/e4109b52-3574-43a8-b95d-33b3494128de/misc/wget-1.17.1/src/http.c:3971
    #8 0x4df57a in retrieve_url /media/bob/e4109b52-3574-43a8-b95d-33b3494128de/misc/wget-1.17.1/src/retr.c:817
    #9 0x40c142 in main /media/bob/e4109b52-3574-43a8-b95d-33b3494128de/misc/wget-1.17.1/src/main.c:1868
    #10 0x7fa89e7b582f in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x2082f)
    #11 0x40e948 in _start (/media/bob/e4109b52-3574-43a8-b95d-33b3494128de/misc/wget-1.17.1/src/wget+0x40e948)

0x61200000bb0f is located 207 bytes inside of 303-byte region [0x61200000ba40,0x61200000bb6f)
allocated by thread T0 here:
    #0 0x7fa89f847e18 in __interceptor_malloc (/usr/lib/x86_64-linux-gnu/libasan.so.3+0xc6e18)
    #1 0x543650 in xmalloc /media/bob/e4109b52-3574-43a8-b95d-33b3494128de/misc/wget-1.17.1/lib/xmalloc.c:41

SUMMARY: AddressSanitizer: negative-size-param (/usr/lib/x86_64-linux-gnu/libasan.so.3+0x47e72) 
==42306==ABORTING
{% endhighlight %}

So long story very short, a very big parameter is passed to memset, which asan catches, however the regular memset happily accept it..

# gdb crash/stack trace

{% highlight text %}
Program received signal SIGSEGV, Segmentation fault.
__memset_avx2 () at ../sysdeps/x86_64/multiarch/memset-avx2.S:161
161	../sysdeps/x86_64/multiarch/memset-avx2.S: No such file or directory.
(gdb) bt
#0  __memset_avx2 () at ../sysdeps/x86_64/multiarch/memset-avx2.S:161
#1  0x0000555555582891 in ?? ()
#2  0x0000555555582e3e in ?? ()
#3  0x0000555555585f32 in ?? ()
#4  0x0000555555575e00 in ?? ()
#5  0x000055555557afac in ?? ()
#6  0x000055555557bd2a in ?? ()
#7  0x0000555555586b64 in ?? ()
#8  0x0000555555561d83 in ?? ()
#9  0x00007ffff6aa8830 in __libc_start_main (main=0x555555560700, argc=2, argv=0x7fffffffe128, init=<optimized out>, fini=<optimized out>, rtld_fini=<optimized out>, stack_end=0x7fffffffe118)
    at ../csu/libc-start.c:291
#10 0x0000555555562019 in ?? ()
(gdb) info registers
rax            0x20202020	538976288
rbx            0xfffffffffffffffc	-4
rcx            0xfffffffffffe876b	-96405
rdx            0xfffffffffffffffc	-4
rsi            0x5555557d776b	93824994867051
rdi            0x5555557ef000	93824994963456
rbp            0x5555557d74d0	0x5555557d74d0
rsp            0x7fffffffd4c8	0x7fffffffd4c8
r8             0x0	0
r9             0x5555557d76bb	93824994866875
r10            0x7fffffffd418	140737488344088
r11            0x0	0
r12            0x0	0
r13            0x71	113
r14            0x5555557cd8f0	93824994826480
r15            0x5555557d776f	93824994867055
rip            0x7ffff6bfa328	0x7ffff6bfa328 <__memset_avx2+392>
eflags         0x10287	[ CF PF SF IF RF ]
cs             0x33	51
ss             0x2b	43
ds             0x0	0
es             0x0	0
fs             0x0	0
gs             0x0	0
{% endhighlight %}
