---
layout: post
title:  "[CVE-2016-7799/7800/7906] 2 ImageMagick bugs and 1 GraphicsMagick bug"
date:   2016-10-02 11:00:00 +0800
categories: security cve
---

## GraphicsMagick CVE-2016-7800

# patch
[https://sourceforge.net/p/graphicsmagick/code/ci/5c7b6d6094a25e99c57f8b18343914ebfd8213ef/](https://sourceforge.net/p/graphicsmagick/code/ci/5c7b6d6094a25e99c57f8b18343914ebfd8213ef/)

# Feedback from the maintainers on oss-security
{% highlight text %}
Today we received a report from Marco Grassi about a heap overflow
in the 8BIM reader. 8BIM is a metadata chunk often attached to JPEG files.

After investigation it was found that there was a small unsigned overflow
leading to a huge size value, which then resulted in a heap overflow
(causing a crash).

We believe that this issue exists in all GraphicsMagick
releases to date (including 1.3.25).
{% endhighlight %}

## ImageMagick CVE-2016-7799

# issue
[https://github.com/ImageMagick/ImageMagick/issues/280](https://github.com/ImageMagick/ImageMagick/issues/280)

# patch
[https://github.com/ImageMagick/ImageMagick/commit/a7bb158b7bedd1449a34432feb3a67c8f1873bfa](https://github.com/ImageMagick/ImageMagick/commit/a7bb158b7bedd1449a34432feb3a67c8f1873bfa)

## ImageMagick CVE-2016-7906

# issue
[https://github.com/ImageMagick/ImageMagick/issues/281](https://github.com/ImageMagick/ImageMagick/issues/281)

# patch
[https://github.com/ImageMagick/ImageMagick/commit/d63a3c5729df59f183e9e110d5d8385d17caaad0](https://github.com/ImageMagick/ImageMagick/commit/d63a3c5729df59f183e9e110d5d8385d17caaad0)
