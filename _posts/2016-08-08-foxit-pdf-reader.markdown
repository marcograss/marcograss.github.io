---
layout: post
title:  "[CVE-2016-6860] Foxit PDF reader memory corruption"
date:   2016-08-08 11:00:00 +0800
categories: security bug cve
---

Recently Foxit published [a new security advisory][sec-adv], among which a bug I disclosed was present, and a new release of their product was pushed to the users.

Like in [another post]({% post_url 2016-07-21-cve-2016-6265-mupdf-uaf %}), this was some kind of collateral bug related to another fuzzing project.

You can find one reproducer here. 

### Reproducer

[foxit1.pdf](/assets/foxit1.pdf)

### How To Test

With a vulnerable version of foxit pdf reader linux:

`MALLOC_CHECK_=3 FoxitReader /path/to/poc/file.pdf`

With a vulnerable version of foxit pdf reader windows, enable page heap to be sure and open

[sec-adv]: https://www.foxitsoftware.com/support/security-bulletins.php
