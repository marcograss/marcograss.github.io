---
layout: post
title:  "Apache Xerces XML parser Use-After-Free"
date:   2016-06-29 11:00:00 +0800
categories: security
---

## TL;DR: I found a XML bug in Apache Xerces by fuzzing, but it turned out that a fix was already pending but not committed.

So in spare time, after giving a try to [expat in another post]({% post_url 2016-06-17-expat-xml-heap-overflow %}), I decided to switch to [Apache Xerces][xerces-website], another xml parser written in C/C++.

I reported the UAF on oss, but Gustavo Grieco (kudos) pointed me out that he already disclosed the same bug, and a fix was [pending but not committed.](https://issues.apache.org/jira/browse/XERCESC-2066) (CVE-2016-2099)

## ASAN Crash

Here you can see the crash (StdInParse is just a tool that takes from stdin the xml):

{% highlight text %}
➜  xml cat xerces_uaf | xerces-c-3.1.3/samples/StdInParse 
=================================================================
==16010==ERROR: AddressSanitizer: heap-use-after-free on address 0xf4a0dfcc at pc 0x0836c7f4 bp 0xfff9a198 sp 0xfff9a188
READ of size 1 at 0xf4a0dfcc thread T0
    #0 0x836c7f3 in xercesc_3_1::ReaderMgr::getLastExtEntityInfo(xercesc_3_1::ReaderMgr::LastExtEntityInfo&) const xercesc/internal/ReaderMgr.cpp:833
    #1 0x83a42d4 in xercesc_3_1::XMLScanner::emitError(xercesc_3_1::XMLErrs::Codes, xercesc_3_1::XMLExcepts::Codes, unsigned short const*, unsigned short const*, unsigned short const*, unsigned short const*) xercesc/internal/XMLScanner.cpp:927
    #2 0x8e40963 in xercesc_3_1::IGXMLScanner::scanDocument(xercesc_3_1::InputSource const&) xercesc/internal/IGXMLScanner.cpp:276
    #3 0x84b4cca in xercesc_3_1::SAXParser::parse(xercesc_3_1::InputSource const&) xercesc/parsers/SAXParser.cpp:575
    #4 0x80533d6 in main src/StdInParse/StdInParse.cpp:186
    #5 0xf6dd5636 in __libc_start_main (/lib32/libc.so.6+0x18636)
    #6 0x80624f1  (/home/bob/VulnResearch/misc/xml/xerces-c-3.1.3/samples/StdInParse+0x80624f1)

0xf4a0dfcc is located 44 bytes inside of 56-byte region [0xf4a0dfa0,0xf4a0dfd8)
freed by thread T0 here:
    #0 0xf7228034 in operator delete(void*) (/usr/lib32/libasan.so.3+0xc5034)
    #1 0x80992df in xercesc_3_1::XMemory::operator delete(void*) xercesc/util/XMemory.cpp:89

previously allocated by thread T0 here:
    #0 0xf72279b4 in operator new(unsigned int) (/usr/lib32/libasan.so.3+0xc49b4)
    #1 0x8357ad9 in xercesc_3_1::MemoryManagerImpl::allocate(unsigned int) xercesc/internal/MemoryManagerImpl.cpp:40
    #2 0x8099042 in xercesc_3_1::XMemory::operator new(unsigned int, xercesc_3_1::MemoryManager*) xercesc/util/XMemory.cpp:68

SUMMARY: AddressSanitizer: heap-use-after-free xercesc/internal/ReaderMgr.cpp:833 in xercesc_3_1::ReaderMgr::getLastExtEntityInfo(xercesc_3_1::ReaderMgr::LastExtEntityInfo&) const
Shadow bytes around the buggy address:
  0x3e941ba0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x3e941bb0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x3e941bc0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x3e941bd0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x3e941be0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
=>0x3e941bf0: fa fa fa fa fd fd fd fd fd[fd]fd fa fa fa fa fa
  0x3e941c00: fd fd fd fd fd fd fd fa fa fa fa fa 00 00 00 00
  0x3e941c10: 00 00 00 fa fa fa fa fa 00 00 00 00 00 00 00 00
  0x3e941c20: fa fa fa fa 00 00 00 00 00 00 00 00 fa fa fa fa
  0x3e941c30: 00 00 00 00 00 00 00 00 fa fa fa fa 00 00 00 00
  0x3e941c40: 00 00 04 fa fa fa fa fa 00 00 00 00 00 00 04 fa
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07 
  Heap left redzone:       fa
  Heap right redzone:      fb
  Freed heap region:       fd
  Stack left redzone:      f1
  Stack mid redzone:       f2
  Stack right redzone:     f3
  Stack partial redzone:   f4
  Stack after return:      f5
  Stack use after scope:   f8
  Global redzone:          f9
  Global init order:       f6
  Poisoned by user:        f7
  Container overflow:      fc
  Array cookie:            ac
  Intra object redzone:    bb
  ASan internal:           fe
  Left alloca redzone:     ca
  Right alloca redzone:    cb
==16010==ABORTING
{% endhighlight %}

And you can download the minimized test case [here](/assets/xerces_uaf)

[xerces-website]: https://xerces.apache.org/xerces-c/
