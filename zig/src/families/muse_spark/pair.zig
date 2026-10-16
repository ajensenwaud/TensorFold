//! A test-only TP=2 collective for both ranks in one process on one GPU (tf-muse m1 with MUSE_PAIR=1): all-gathers
//! by device copies between two host threads that meet at a spin barrier. Proves the two-rank arithmetic (cuts,
//! gathers, rank-order sums) where NCCL refuses two ranks on one device; synchronous, eager only, no speed meaning.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const C = tp.Collective;
const K = tp.collective;

pub const Pair = struct {
    d: *const cuda.Driver,
    arrived: std.atomic.Value(u32) = .init(0),
    gen: std.atomic.Value(u32) = .init(0),
    aborted: std.atomic.Value(bool) = .init(false),
    dst: [2]u64 = .{ 0, 0 },

    fn barrier(p: *Pair) K.Error!void {
        const g = p.gen.load(.acquire);
        if (p.arrived.fetchAdd(1, .acq_rel) == 1) {
            p.arrived.store(0, .release);
            _ = p.gen.fetchAdd(1, .release);
            return;
        }
        while (p.gen.load(.acquire) == g) {
            if (p.aborted.load(.acquire)) return error.Aborted;
            std.atomic.spinLoopHint();
        }
    }

    pub fn abort(p: *Pair) void {
        p.aborted.store(true, .release);
    }
};

pub const Rank = struct {
    pair: *Pair,
    rank: u32,

    fn of(ptr: *anyopaque) *Rank {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn collective(r: *Rank) C {
        return .{ .ptr = r, .vtable = &vtable };
    }

    const vtable: C.VTable = .{ .kind = .host, .rank = rankFn, .world = worldFn, .all_reduce = allReduce, .all_gather = allGather, .exchange = exchange, .send = sendFn, .recv = recvFn, .barrier = barrierFn, .check = check, .abort = abortFn };

    fn rankFn(ptr: *anyopaque) u32 {
        return of(ptr).rank;
    }
    fn worldFn(_: *anyopaque) u32 {
        return 2;
    }
    /// recv[2 * count] <- both ranks' send[count]: each rank waits for its stream, publishes recv, copies its slice
    /// into both ranks' recv, and waits for the other's copies.
    fn allGather(ptr: *anyopaque, send: K.DevicePtr, recv: K.DevicePtr, count: usize, t: tp.DType, stream: K.Stream) K.Error!void {
        const r = of(ptr);
        const p = r.pair;
        const d = p.d;
        d.check(d.api.cuStreamSynchronize(stream), "pair sync") catch return error.BackendFailed;
        p.dst[r.rank] = recv;
        try p.barrier();
        const bytes = count * t.size();
        for (p.dst) |dst| d.check(d.api.cuMemcpyDtoD_v2(dst + r.rank * bytes, send, bytes), "pair copy") catch return error.BackendFailed;
        try p.barrier();
    }
    fn allReduce(_: *anyopaque, _: K.DevicePtr, _: K.DevicePtr, _: usize, _: tp.DType, _: tp.Op, _: K.Stream) K.Error!void {
        return error.Unsupported;
    }
    fn exchange(_: *anyopaque, _: K.DevicePtr, _: K.DevicePtr, _: usize, _: tp.DType, _: u32, _: K.Stream) K.Error!void {
        return error.Unsupported;
    }
    fn sendFn(_: *anyopaque, _: K.DevicePtr, _: usize, _: tp.DType, _: u32, _: K.Stream) K.Error!void {
        return error.Unsupported;
    }
    fn recvFn(_: *anyopaque, _: K.DevicePtr, _: usize, _: tp.DType, _: u32, _: K.Stream) K.Error!void {
        return error.Unsupported;
    }
    fn barrierFn(ptr: *anyopaque) K.Error!void {
        return of(ptr).pair.barrier();
    }
    fn check(ptr: *anyopaque) K.Error!void {
        if (of(ptr).pair.aborted.load(.acquire)) return error.Aborted;
    }
    fn abortFn(ptr: *anyopaque) void {
        of(ptr).pair.abort();
    }
};
