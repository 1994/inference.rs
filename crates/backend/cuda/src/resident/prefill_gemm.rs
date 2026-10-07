//! Prompt GEMM: quantized weights are decoded in tiles, BF16 MMA accumulates in F32.
#[cutile::module]
pub(crate) mod gemm {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, Shape_3, StoreTileAtCurrentBlock, Tensor_1,
        Tensor_2, Tile_2, UnpackF4e2m1fnx2Tile, bf16, constant, convert_tile, f4e2m1fnx2, f8e4m3fn,
        get_tile_block_id, mmaf,
    };

    #[cutile::entry()]
    fn dense<const K: i32>(
        out: &mut Tensor<f32, { [32, 64] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<bf16, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![32, 64]);
        let wp = weight.partition(shape![64, 64]);
        let mut acc: Tile<f32, { [32, 64] }> = constant(0.0f32, shape![32, 64]);
        for k in 0i32..((K + 63) / 64) {
            let x: Tile<bf16, { [32, 64] }> = convert_tile(xp.load([pid.0, k]));
            acc = mmaf(x, wp.load([pid.1, k]).transpose(), acc);
        }
        out.store(acc);
    }

    #[cutile::entry()]
    fn fp8<const K: i32>(
        out: &mut Tensor<f32, { [32, 64] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<f8e4m3fn, { [-1, K] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![32, 64]);
        let wp = weight.partition(shape![64, 64]);
        let mut acc: Tile<f32, { [32, 64] }> = constant(0.0f32, shape![32, 64]);
        for k in 0i32..((K + 63) / 64) {
            let x: Tile<bf16, { [32, 64] }> = convert_tile(xp.load([pid.0, k]));
            let w: Tile<bf16, { [64, 64] }> = convert_tile(wp.load([pid.1, k]));
            acc = mmaf(x, w.transpose(), acc);
        }
        let scales = scale
            .partition(shape![64])
            .load([pid.1])
            .reshape(shape![1, 64])
            .broadcast(shape![32, 64]);
        out.store(acc * scales);
    }

    #[cutile::entry()]
    fn nvfp4<const K: i32>(
        out: &mut Tensor<f32, { [32, 64] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scale: &Tensor<f8e4m3fn, { [-1, -1] }>,
        inverse_global: f32,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![32, 64]);
        let wp = weight.partition(shape![64, 32]);
        let sp = scale.partition(shape![64, 4]);
        let mut acc: Tile<f32, { [32, 64] }> = constant(0.0f32, shape![32, 64]);
        for k in 0i32..((K + 63) / 64) {
            let x: Tile<bf16, { [32, 64] }> = convert_tile(xp.load([pid.0, k]));
            let w: Tile<f32, { [64, 64] }> =
                convert_tile(wp.load([pid.1, k]).unpack(shape![64, 64]));
            let s: Tile<f32, { [64, 4] }> = convert_tile(sp.load([pid.1, k]));
            let s = s
                .reshape(shape![64, 4, 1])
                .broadcast(shape![64, 4, 16])
                .reshape(shape![64, 64]);
            let w: Tile<bf16, { [64, 64] }> = convert_tile(w * s);
            acc = mmaf(x, w.transpose(), acc);
        }
        out.store(acc * inverse_global.broadcast(shape![32, 64]));
    }

    /// Split-K variant: `partials` is `[splits * 32, N]` tiled `[32, 64]`, so block row
    /// `pid.0` is the split index and each block accumulates only its K-tile window.
    /// Partials stay unscaled; `reduce_split` sums the windows and applies the scale.
    #[cutile::entry()]
    fn nvfp4_split<const K: i32>(
        out: &mut Tensor<f32, { [32, 64] }>,
        input: &Tensor<f32, { [-1, K] }>,
        weight: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        scale: &Tensor<f8e4m3fn, { [-1, -1] }>,
        k_tiles: i32,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![32, 64]);
        let wp = weight.partition(shape![64, 32]);
        let sp = scale.partition(shape![64, 4]);
        let total = (K + 63) / 64;
        let k0 = pid.0 * k_tiles;
        let mut k1 = k0 + k_tiles;
        if k1 > total {
            k1 = total;
        }
        let mut acc: Tile<f32, { [32, 64] }> = constant(0.0f32, shape![32, 64]);
        for k in k0..k1 {
            let x: Tile<bf16, { [32, 64] }> = convert_tile(xp.load([0i32, k]));
            let w: Tile<f32, { [64, 64] }> =
                convert_tile(wp.load([pid.1, k]).unpack(shape![64, 64]));
            let s: Tile<f32, { [64, 4] }> = convert_tile(sp.load([pid.1, k]));
            let s = s
                .reshape(shape![64, 4, 1])
                .broadcast(shape![64, 4, 16])
                .reshape(shape![64, 64]);
            let w: Tile<bf16, { [64, 64] }> = convert_tile(w * s);
            acc = mmaf(x, w.transpose(), acc);
        }
        out.store(acc);
    }

    /// Sums the `[splits, 32, N]` partial windows in split order and scales the result.
    #[cutile::entry()]
    fn reduce_split(
        out: &mut Tensor<f32, { [32, 64] }>,
        partials: &Tensor<f32, { [-1, -1] }>,
        inverse_global: f32,
        splits: i32,
    ) {
        let pid = get_tile_block_id();
        let pp = partials.partition(shape![32, 64]);
        let mut acc: Tile<f32, { [32, 64] }> = constant(0.0f32, shape![32, 64]);
        for s in 0i32..splits {
            acc = acc + pp.load([s, pid.1]);
        }
        out.store(acc * inverse_global.broadcast(shape![32, 64]));
    }
}
