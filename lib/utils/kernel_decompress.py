import torch


#@torch.compile
def decode_compressed(L, S, R, V, m, k, compressed, expanded_lut):
    if compressed.dtype != torch.uint16:
        compressed = compressed.view(torch.uint16)

    assert compressed.shape == (R * m * k // 16, )

    BITS_PER_BLOCK = R * 16 * 16  # R bits * f16 mma tile A size

    # unswizzle interleaved blocks

    BLOCK_SIZE = 16 * 16

    BITS_PER_BLOCK = R * 16 * 16  # R bits * f16 mma tile A size

    compressed = (compressed.view(torch.uint8).reshape(
        m // 16 // 2, k // 16 // 2, BLOCK_SIZE // 8, 2, 2,
        R).permute(0, -2, 1, -3, 2, -1).flip(
            (-1, )).reshape(m // 16, k // 16, BITS_PER_BLOCK // 16, 2).flip(
                (-1, )).view(torch.uint16).reshape(m // 16, k // 16,
                                                   BITS_PER_BLOCK // 16))
    # decode block

    assert L <= 16

    blocked = compressed.reshape(R * m * k // BITS_PER_BLOCK,
                                 BITS_PER_BLOCK // 16, 1)
    blocked_roll = torch.roll(blocked.to(torch.int32), -1,
                              -2).to(blocked.dtype)
    blocked32 = torch.cat((blocked_roll, blocked),
                          dim=-1).reshape(blocked.shape[0],
                                          -1).contiguous().view(torch.uint32)
    # blocked32 is 16bits[-1]||16bits[0] 16bits[0]||16bits[1] ... 16bits[-2]||16bits[-1]

    expanded32 = blocked32.reshape(*blocked32.shape,
                                   1).expand(*blocked32.shape,
                                             16).view(torch.int32)
    shifts = (torch.arange(0, 16, dtype=torch.int32,
                           device=blocked.device)).to(torch.int32).reshape(
                               1, 1, -1).expand(expanded32.shape)
    shifted = expanded32 >> (16 - shifts)
    indices = torch.bitwise_and(
        shifted.reshape(shifted.shape[0], -1)[:, 16 - L::R << V], (1 << L) - 1)

    # decode lut
    mma_swizzled = expanded_lut[indices]

    # deswizzle m16n8k16 mma pattern
    decompressed = (mma_swizzled.reshape(m // 16, k // 16, 16, 16).reshape(
        m // 16, k // 16, 8, 4, 2, 2, 2).permute(0, -2, 2, 1, -3, 3,
                                                 -1).reshape(m, k))
    return decompressed
