---
layout: post
title:  "Remotely pwning iOS via WiFi and escaping the Sandbox (OLD UNPUBLISHED 2016 BLOGPOST) "
date:   2016-06-17 18:00:00 +0800
categories: security apple ios wifi
---

## Notes

This is a post I wrote in June 2016 but never published, so quite a lot changed, and countless iOS releases. Anyway maybe someone is interested in reading it. I will post it unedited even if there has been some amazing research about this topic in those 5 years, which you can find online. Everything mentioned here has been fixed years ago.

# Abstract

In this document we will exploit a iOS device remotely via WiFi without any user interaction, completely bypassing the iOS sandbox and compromise the user device in various points as a demo of the danger of such exploit chain.

The victim will only have to join our WiFi network, and then the device will be compromised without any user interaction, bypassing all iOS mitigations and sandboxes.

# Outline
1. Introduction
2. WiFi access points, captive portals, WebSheet and initial code execution
3. Entitlements, NSXPC services, other logic bugs and finally out of the Sandbox!

## Introduction

Especially in recent years mobile devices really became rich of features, I would say even more than desktop. But as we will see, more features often imply more attack surface and more dangers for the users.

Some months ago I started exploring alternative attack surfaces on iOS devices, especially focusing on WiFi.

Last year there was also another very exciting [remote pwn over WiFi by Mark Dowd][dowd-airdrop].

I'm pretty sure all iOS users are familiar when going to Starbucks or any other public place with WiFi where you need to login, after joining a WiFi network iOS will open a window with a web page where to do login.

How does this work? Let's see... Because our story will start exactly from there!

## WiFi access points, captive portals, WebSheet and initial code execution

I rarely use iOS devices for personal use myself, but some months ago noticed that when I logged in a WiFi network that required to log in, I was prompted with a customized web page inside of an application which was not MobileSafari.

This application is called WebSheet and it's a system application on all iOS devices. It will automatically popup if our WiFi device have a very particular behavior.

Lot of questions and doubts started going in and out of my mind:

* What is this application?
* How a WiFi provider render its content inside of it?
* Which kind of sandbox this application that it's rendering untrusted web content is running into?

As some readers may know, MobileSafari sandbox is strong, so potentially facing a different sandbox was already very promising.

So I started reversing and experimenting and I happily found interesting properties of this WebSheet application.

But first, let's see how to gain control of the web content rendered inside this application on our own malicious network.

When you connect any WiFi network, your iOS device will issue a web request to a predefined URL, to see if it's reachable. This URL is `http://captive.apple.com/hotspot-detect.html`.

If the original apple website is reached the HTML content will be exactly:

`<HTML><HEAD><TITLE>Success</TITLE></HEAD><BODY>Success</BODY></HTML>`

If a captive portal on the wifi catch this request (it's http so no problem intercepting this), can return any html content they want.

Which is exactly what we are going to do.

We will setup a malicious accesspoint leveraging a linux machine, a wifi dongle and hostapd.

This malicious access point will run mitmproxy which will intercept all web request going back and forward from the iOS device.

When we detect that a request is made to `http://captive.apple.com/hotspot-detect.html` we will just return a malicious page with our `WebKit` exploit, which will be rendered on the device, and code execution inside WebSheet will be achieved! Perfect!

The WebKit vulnerability is out of the scope of this post, the important fact is that now we have code execution inside WebSheet on the iOS device, remotely and without any user interaction when the user joined our Wifi Network!

A small chunk of code modified to don't include the WebKit exploit from the mitmproxy code:

{% highlight python %}
def response(context, flow):
    flow.response.content = "<HTML><HEAD><TITLE>derp</TITLE><script>alert('exploiting webkit here');</script></HEAD><BODY><h1>derp</h1></BODY></HTML>"
    flow.response.headers['Content-Length'] = str(len(flow.response.content))
{% endhighlight %}

Let's now examine the WebSheet sandbox for a moment...

I'm not gonna post here the sandbox profile of WebSheet, but basically this sandbox is very restrictive, it's almost like the MobileSafari WebProcess sandbox, probably less privileged then a Container application, so we have code execution in a very restrictive environment unfortunately...

Notice the big weakness of WebSheet.. it doesn't use a split process model like MobileSafari to sandbox the renderer process, nor it uses the new WKWebView, which in fact runs web content in a separate process.. So if we use a web bug, we get code execution in the context of WebSheet.

So let's summarize what bugs we used now:

* a logic bug giving us the ability to automatically render webcontent in WebSheet if we have wifi control

* a webkit bug to get code execution

So now we have code execution inside WebSheet, but the sandbox is very restrictive, what to do next?

ESCAPE THE SANDBOX of course!

# Entitlements, NSXPC services, other logic bugs and finally out of the Sandbox!

Maybe lot of the readers are familiar with the concept of entitlements. Basically it's just extra capabilities given to certain binaries and signed like all the code, so they are legit.

In order to understand what I could do after getting initial code execution inside WebSheet I examined which entitlements the app have.

{% highlight xml %}
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>com.apple.TextInput.rdt.me</key>
  <true/>
  <key>com.apple.UIKit.vends-view-services</key>
  <true/>
  <key>com.apple.captive.private</key>
  <true/>
  <key>com.apple.managedconfiguration.profiled-access</key>
  <true/>
  <key>com.apple.private.necp.match</key>
  <true/>
  <key>com.apple.security.exception.mach-lookup.global-name</key>
  <array>
    <string>com.apple.mobilesafari-settings</string>
  </array>
  <key>com.apple.springboard.opensensitiveurl</key>
  <true/>
  <key>com.apple.springboard.openurlinbackground</key>
  <true/>
  <key>keychain-access-groups</key>
  <array>
    <string>com.apple.cfnetwork</string>
    <string>com.apple.mobilesafari</string>
    <string>com.apple.certificates</string>
    <string>com.apple.safari.credit-cards</string>
  </array>
  <key>seatbelt-profiles</key>
  <array>
    <string>WebSheet</string>
  </array>
</dict>
</plist>
{% endhighlight %}

The one I will use are: `com.apple.managedconfiguration.profiled-access` and `com.apple.springboard.opensensitiveurl` for our sandbox escape :)

I will use the first to install a configuration profile silently, in this way we can install our own Root CA to intercept and edit the HTTPS traffic, not only the HTTP.

I will also use it to demonstrate how dangerous it can be, I will install an additional fake app on the iOS device, I will change the APN settings and set a proxy, essentially compromising forever and dramatically the iOS device.. Anyway all those are side effects.. And they are all capabilities provided from configuration profile.

The problem is that we can install a configuration profile silently and without user interaction with that entitlement.

Usually when installing a configuration profile, a user is prompted with this UI, but this time it will not, because we have the entitlement.

Let's just focus on getting code execution outside of the sandbox now.

We will execute this code as a ROP payload, thanks to our RCE webkit bug:
{% highlight c %}
void install_profile() {
    NSString *base64profile = @"OMITTEDPROFILE_BASE64";

    NSData *profileData = [[NSData alloc] initWithBase64EncodedString:base64profile options:0];

    MCProfileConnection *profileConnection = [MCProfileConnection sharedConnection];
    NSLog(@"profile conn %@", profileConnection);

    id error;
    NSString *profileId = [profileConnection installProfileData:profileData outError:&error];

    NSLog(@"installed profile id %@ err %@", profileId, error);
}
{% endhighlight %}

This code will speak to a system deamon via xpc/NSXPC to install our provided configuration profile silently, very convenient!

So as you can see now the profile is installed. With our Root CA.

Now we will use another bug to laterally move to another Application which is not sandboxed. And reuse the same WebKit RCE bug to get code execution out of the sandbox!

Always in ROP we will execute the following:

{% highlight c %}
[[LSApplicationWorkspace defaultWorkspace] openSensitiveURL:@"legacy-diagnostics:" withOptions:0];
{% endhighlight %}

This will popup a Diagnostic application, which will issue a https request, which we will mitm again, and serve the same exploit, finally gaining remote code execution out of the sandbox!

This iOS Feedback is running unsandboxed on the system.

We can then again leverage mitmproxy to serve the webkit exploit like before. The request is https, but remember, WE INSTALLED A ROOT CA on the device, so we can man in the middle this request now!

{% highlight python %}
def response(context, flow):
    if flow.request.path.endswith('ios/LegalText/1.2') or flow.request.path.endswith('GetOptinText'):
        flow.response.content = "<script>alert('exploit again');</script>"
        flow.response.headers['Content-Length'] = str(len(flow.response.content))
        if 'Content-Encoding' in flow.response.headers: del flow.response.headers['Content-Encoding']
{% endhighlight %}

The javascript content we will provide will end up again in a WebView inside this application, which we can exploit again for RCE.

Aand... Finally we have code execution outside of the Sandbox as user mobile :)

From there you can open plenty of IOKit driver and attack them, potentially making a remote silent jailbreak over wifi (lol).

So let's summarize which bug we used for the sandbox escape:

* We leveraged the overprivilege of WebSheet, and the private API to silently install a configuration profile

* We leveraged the URL scheme legacy-diagnostics to pivot to a unsandboxed app, together with open sensitive url entitlement, again WebSheet is overprivileged

* The privileged app is not doing certificate pinning, so we can mitm the https traffic thanks to our rogue Root CA that we installed.

* The same webkit bug again.

Not bad for a single memory corruption bug and a bunch of logic bugs!

[dowd-airdrop]: http://2015.ruxcon.org.au/assets/2015/slides/ruxcon-2016-dowd.pptx
