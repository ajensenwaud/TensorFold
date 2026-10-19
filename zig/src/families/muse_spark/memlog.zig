//! A rank's host memory for the recipe's memory accounting and its host-side levers (each off unless set):
//!   TF_MUSE_MEMLOG=1      a line per load stage and per graph capture: MemAvailable, this process's RSS, swap and
//!                         pinned pages, plus the caller's device numbers (arenas' handed-out and held bytes). Load
//!                         stages carry a wall-clock time and wait 1.5 s, so a 1 s nvidia-smi sampler outside sees
//!                         each stage's device total per process.
//!   TF_MUSE_TRIM=1        malloc_trim(0) after load and after each graph capture: freed host pages (load staging,
//!                         captured graph templates) go back to the OS instead of sitting in the heap or in swap.
//!   TF_MUSE_DROP_CACHE=1  after each layer's upload, the pack's mapped pages leave this process (MADV_DONTNEED on the
//!                         read-only mapping: a later read refaults) and the page cache (POSIX_FADV_DONTNEED): the load
//!                         no longer holds a second, host copy of what it has uploaded.
//! None of them changes a device byte.

const std = @import("std");

extern "c" fn malloc_trim(pad: usize) c_int;
extern "c" fn gettimeofday(tv: *[2]c_long, tz: ?*anyopaque) c_int;
extern "c" fn usleep(us: c_uint) c_int;
extern "c" fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) c_int;
extern "c" fn madvise(addr: *anyopaque, len: usize, advice: c_int) c_int;
extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fread(ptr: [*]u8, size: usize, n: usize, f: *anyopaque) usize;
extern "c" fn fclose(f: *anyopaque) c_int;

const MADV_DONTNEED: c_int = 4;
const POSIX_FADV_DONTNEED: c_int = 4;

fn flag(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

pub fn on() bool {
    return flag("TF_MUSE_MEMLOG");
}
pub fn trimOn() bool {
    return flag("TF_MUSE_TRIM");
}
pub fn dropOn() bool {
    return flag("TF_MUSE_DROP_CACHE");
}

/// `key:` of a /proc file in kB (0 when absent).
pub fn procKb(path: [*:0]const u8, key: []const u8) usize {
    var buf: [16384]u8 = undefined;
    const f = fopen(path, "r") orelse return 0;
    defer _ = fclose(f);
    const n = fread(&buf, 1, buf.len, f);
    var it = std.mem.splitScalar(u8, buf[0..n], '\n');
    while (it.next()) |ln| {
        if (ln.len > key.len and std.mem.startsWith(u8, ln, key) and ln[key.len] == ':') {
            var t = std.mem.tokenizeAny(u8, ln[key.len + 1 ..], " \t");
            return std.fmt.parseInt(usize, t.next() orelse return 0, 10) catch 0;
        }
    }
    return 0;
}

pub fn mib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 20);
}

fn line(rank: usize, what: []const u8, extra: []const u8) void {
    var tv: [2]c_long = .{ 0, 0 };
    _ = gettimeofday(&tv, null);
    std.debug.print("[memlog] rank {d} t={d}.{d:0>3} {s}: MemAvailable {d} MiB, rss {d} swap {d} pinned {d} MiB; {s}\n", .{
        rank,                                    tv[0],                                   @divTrunc(tv[1], 1000), what,
        procKb("/proc/meminfo", "MemAvailable") >> 10, procKb("/proc/self/status", "VmRSS") >> 10, procKb("/proc/self/status", "VmSwap") >> 10, procKb("/proc/self/status", "VmPin") >> 10,
        extra,
    });
}

/// A load stage (TF_MUSE_MEMLOG): the line, then 1.5 s for the sampler outside.
pub fn stage(rank: usize, what: []const u8, comptime fmt: []const u8, args: anytype) void {
    if (!on()) return;
    var b: [512]u8 = undefined;
    line(rank, what, std.fmt.bufPrint(&b, fmt, args) catch "(long)");
    _ = usleep(1_500_000);
}

/// A run-time event (a graph capture): the line, no wait.
pub fn note(rank: usize, what: []const u8, comptime fmt: []const u8, args: anytype) void {
    if (!on()) return;
    var b: [512]u8 = undefined;
    line(rank, what, std.fmt.bufPrint(&b, fmt, args) catch "(long)");
}

/// TF_MUSE_TRIM: freed heap pages back to the OS (and out of swap).
pub fn trim(rank: usize, what: []const u8) void {
    if (!trimOn()) return;
    const before = procKb("/proc/self/status", "VmRSS") + procKb("/proc/self/status", "VmSwap");
    _ = malloc_trim(0);
    const after = procKb("/proc/self/status", "VmRSS") + procKb("/proc/self/status", "VmSwap");
    if (on()) {
        var b: [128]u8 = undefined;
        line(rank, what, std.fmt.bufPrint(&b, "malloc_trim: rss + swap {d} -> {d} MiB", .{ before >> 10, after >> 10 }) catch "");
    }
}

/// TF_MUSE_DROP_CACHE: a read-only mapped file's pages out of this process and out of the page cache.
pub fn dropFile(path: [:0]const u8, map: []const u8) void {
    if (map.len > 0) _ = madvise(@ptrCast(@constCast(map.ptr)), map.len, MADV_DONTNEED);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return;
    defer _ = std.c.close(fd);
    _ = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
}
