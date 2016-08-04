---
layout: post
title:  "[CVE-2016-6583] libtidy global buffer overflow"
date:   2016-08-03 11:00:00 +0800
categories: bug
---

Recently I got interested into a html parsing library called tidy, which is also used in Apple products apparently.

Sometimes when developers doesn't usually run stuff on asan or valgrind, you just hit bugs by running binaries.

Which is exactly what happened. So I just reported the bug as non security one, to being able to run the lib with asan.

Turns out that mitre think that maybe there is the possibility that this global bof can be reached from applications using this code as a library (libtidy).

I still have to check this because I'm in Vegas for BH/Defcon so I cannot actually check it right now, but anyway, here's the asan report, triggered in master and latest stable, and github issue.

# github issue
[https://github.com/htacg/tidy-html5/issues/443](https://github.com/htacg/tidy-html5/issues/443)

# potential fix (to test)

[https://github.com/htacg/tidy-html5/pull/445](https://github.com/htacg/tidy-html5/pull/445)

# crash report

{% highlight text %}
==13070==ERROR: AddressSanitizer: global-buffer-overflow on address 0x082b12e6 at pc 0xf7177cf6 bp 0xffc16078 sp 0xffc15c48
READ of size 7 at 0x082b12e6 thread T0
#0 0xf7177cf5 (/usr/lib/i386-linux-gnu/libasan.so.3+0x32cf5)
#1 0x80adab9 in tidySetLanguage /home/bob/VulnResearch/misc/xml/tidy-html5-5.2.0/src/language.c:705
#2 0x805339b in main /home/bob/VulnResearch/misc/xml/tidy-html5-5.2.0/console/tidy.c:1539
#3 0xf6fa7636 in __libc_start_main (/lib/i386-linux-gnu/libc.so.6+0x18636)
#4 0x8058553 (/home/bob/VulnResearch/misc/xml/tidy-html5-5.2.0/build/cmake/tidy+0x8058553)

0x082b12e6 is located 58 bytes to the left of global variable 'tidyLanguages' defined in '/home/bob/VulnResearch/misc/xml/tidy-html5-5.2.0/src/language.c:38:26' (0x82b1320) of size 36
0x082b12e6 is located 0 bytes to the right of global variable 'result' defined in '/home/bob/VulnResearch/misc/xml/tidy-html5-5.2.0/src/language.c:603:17' (0x82b12e0) of size 6
SUMMARY: AddressSanitizer: global-buffer-overflow (/usr/lib/i386-linux-gnu/libasan.so.3+0x32cf5) 
Shadow bytes around the buggy address:
0x21056200: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0x21056210: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0x21056220: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0x21056230: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0x21056240: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
=>0x21056250: 00 00 00 00 00 00 00 00 00 00 00 00[06]f9 f9 f9
0x21056260: f9 f9 f9 f9 00 00 00 00 04 f9 f9 f9 f9 f9 f9 f9
0x21056270: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0x21056280: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0x21056290: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
0x210562a0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
Shadow byte legend (one shadow byte represents 8 application bytes):
Addressable: 00
Partially addressable: 01 02 03 04 05 06 07 
Heap left redzone: fa
Heap right redzone: fb
Freed heap region: fd
Stack left redzone: f1
Stack mid redzone: f2
Stack right redzone: f3
Stack partial redzone: f4
Stack after return: f5
Stack use after scope: f8
Global redzone: f9
Global init order: f6
Poisoned by user: f7
Container overflow: fc
Array cookie: ac
Intra object redzone: bb
ASan internal: fe
Left alloca redzone: ca
Right alloca redzone: cb
==13070==ABORTING
{% endhighlight %}
