//! Checkpoint-declared dynamic per-token E4M3 activation quantization.
#[cutile::module]
pub(crate) mod kernels {
    use cutile::core::{
        BroadcastScalar, Reshape, Shape_1, Shape_2, StoreTileAtCurrentBlock, Tensor_1, Tensor_2,
        Tile_1, Tile_2, absf, constant, convert_tile, f8e4m3fn, get_tile_block_id, max_tile,
        min_tile, mmaf, reduce_max,
    };

    #[cutile::entry()]
    fn quantize<const K: i32, const PAD: i32>(
        output: &mut Tensor<f8e4m3fn, { [1, PAD] }>,
        scale: &mut Tensor<f32, { [1, 1] }>,
        input: &Tensor<f32, { [-1, K] }>,
    ) {
        let pid = get_tile_block_id();
        let x = input.partition(shape![1, PAD]).load([pid.0, 0i32]);
        let maximum: Tile<f32, { [1] }> = reduce_max(absf(x), 1i32);
        let maximum = maximum.reshape(shape![1, 1]);
        let s = max_tile(
            maximum / 448.0f32.broadcast(shape![1, 1]),
            1.0e-12f32.broadcast(shape![1, 1]),
        );
        let x = x / s.broadcast(shape![1, PAD]);
        let x = min_tile(
            max_tile(x, (-448.0f32).broadcast(shape![1, PAD])),
            448.0f32.broadcast(shape![1, PAD]),
        );
        let packed: Tile<f8e4m3fn, { [1, PAD] }> = convert_tile(x);
        output.store(packed);
        scale.store(s);
    }

    #[cutile::entry()]
    fn matmul<const K: i32>(
        output: &mut Tensor<f32, { [16, 64] }>,
        input: &Tensor<f8e4m3fn, { [-1, K] }>,
        weight: &Tensor<f8e4m3fn, { [-1, K] }>,
        input_scale: &Tensor<f32, { [-1, 1] }>,
        weight_scale: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let xp = input.partition(shape![16, 128]);
        let wp = weight.partition(shape![64, 128]);
        let mut acc: Tile<f32, { [16, 64] }> = constant(0.0f32, shape![16, 64]);
        for k in 0i32..((K + 127) / 128) {
            acc = mmaf(xp.load([pid.0, k]), wp.load([pid.1, k]).transpose(), acc);
        }
        let xs = input_scale
            .partition(shape![16, 1])
            .load([pid.0, 0i32])
            .broadcast(shape![16, 64]);
        let ws = weight_scale
            .partition(shape![64])
            .load([pid.1])
            .reshape(shape![1, 64])
            .broadcast(shape![16, 64]);
        output.store(acc * xs * ws);
    }
}

#[cfg(test)]
mod tests;
