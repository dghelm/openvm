use std::{ffi::c_void, sync::Arc};

use openvm_circuit::{
    arch::{MemoryConfig, ADDR_SPACE_OFFSET},
    system::memory::{merkle::MemoryMerkleCols, TimestampedEquipartition},
    utils::next_power_of_two_or_zero,
};
use openvm_hip_backend::{base::DeviceMatrix, prelude::F, prover_backend::HipBackend};
use openvm_hip_common::{
    copy::{hip_memcpy, MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::{default_stream_wait, hipStreamPerThread, HipEvent, HipStream},
};
use openvm_stark_backend::{
    p3_maybe_rayon::prelude::{IntoParallelIterator, ParallelIterator},
    p3_util::log2_ceil_usize,
    prover::types::AirProvingContext,
};
use p3_field::FieldAlgebra;

use super::{poseidon2::SharedBuffer, Poseidon2PeripheryChipHIP, DIGEST_WIDTH};

pub mod hip;
use hip::merkle_tree::*;

type H = [F; DIGEST_WIDTH];
pub const TIMESTAMPED_BLOCK_WIDTH: usize = 11;

/// A Merkle subtree stored in a single flat buffer, combining a vertical path and a heap-ordered
/// binary tree.
///
/// Memory layout:
/// - The first `path_len` elements form a vertical path (one node per level), used when the actual
///   size is smaller than the max size.
/// - The remaining elements store the subtree nodes in heap-order (breadth-first), with `size`
///   leaves and `2 * size - 1` total nodes.
///
/// The call of filling the buffer is done async on the new stream. Option<HipEvent> is used to
/// wait for the completion.
pub struct MemoryMerkleSubTree {
    pub stream: Arc<HipStream>,
    pub event: Option<HipEvent>,
    pub buf: DeviceBuffer<H>,
    pub height: usize,
    pub path_len: usize,
}

impl MemoryMerkleSubTree {
    /// Constructs a new Merkle subtree with a vertical path and heap-ordered tree.
    /// The buffer is sized based on the actual address space and the maximum size.
    ///
    /// `addr_space_size` is the number of leaf digest nodes necessary for this address space. The
    /// `max_size` is the number of leaf digest nodes in the full balanced tree dictated by
    /// `addr_space_height` from the `MemoryConfig`.
    pub fn new(addr_space_size: usize, max_size: usize) -> Self {
        assert!(
            max_size.is_power_of_two(),
            "Max address space size must be a power of two"
        );
        let size = next_power_of_two_or_zero(addr_space_size);
        if addr_space_size == 0 {
            let mut res = MemoryMerkleSubTree::dummy();
            res.height = log2_ceil_usize(max_size);
            return res;
        }
        let height = log2_ceil_usize(size);
        let path_len = log2_ceil_usize(max_size).checked_sub(height).unwrap();
        tracing::debug!(
            "Creating a subtree buffer, size is {} (addr space size is {})",
            path_len + (2 * size - 1),
            addr_space_size
        );
        let buf = DeviceBuffer::<H>::with_capacity(path_len + (2 * size - 1));

        let created_buffer_event = HipEvent::new().unwrap();
        unsafe {
            created_buffer_event.record(hipStreamPerThread).unwrap();
        }

        let stream = Arc::new(HipStream::new().unwrap());
        stream.wait(&created_buffer_event).unwrap();
        Self {
            stream,
            event: None,
            height,
            buf,
            path_len,
        }
    }

    pub fn dummy() -> Self {
        Self {
            stream: Arc::new(HipStream::new().unwrap()),
            event: None,
            height: 0,
            buf: DeviceBuffer::new(),
            path_len: 0,
        }
    }

    /// Asynchronously builds the Merkle subtree on its dedicated HIP stream.
    /// Also reconstructs the vertical path if `path_len > 0`, and records a completion event.
    ///
    /// Here `addr_space_idx` is the address space _shifted_ by ADDR_SPACE_OFFSET = 1
    pub fn build_async(
        &mut self,
        d_data: &DeviceBuffer<u8>,
        addr_space_idx: usize,
        zero_hash: &DeviceBuffer<H>,
    ) {
        let event = HipEvent::new().unwrap();
        if self.buf.is_empty() {
            // TODO not really async in this branch is it
            self.buf = DeviceBuffer::with_capacity(1);
            unsafe {
                hip_memcpy::<true, true>(
                    self.buf.as_mut_raw_ptr(),
                    zero_hash.as_ptr().add(self.height) as *mut c_void,
                    size_of::<H>(),
                )
                .unwrap();
                event.record(hipStreamPerThread).unwrap();
            }
        } else {
            unsafe {
                build_merkle_subtree(
                    d_data,
                    1 << self.height,
                    &self.buf,
                    self.path_len,
                    addr_space_idx as u32,
                    self.stream.as_raw(),
                )
                .unwrap();

                if self.path_len > 0 {
                    restore_merkle_subtree_path(
                        &self.buf,
                        zero_hash,
                        self.path_len,
                        self.height + self.path_len,
                        self.stream.as_raw(),
                    )
                    .unwrap();
                }
                event.record(self.stream.as_raw()).unwrap();
            }
        }
        self.event = Some(event);
    }

    /// Returns the bounds [start, end) of the layer at the given depth.
    /// These bounds correspond to the indices of the layer in the buffer.
    /// depth: 0 = root, 1 = root's children, ..., height-1 = leaves
    pub fn layer_bounds(&self, depth: usize) -> (usize, usize) {
        let global_height = self.height + self.path_len;
        assert!(
            depth < global_height,
            "Depth {} out of bounds for height {}",
            depth,
            global_height
        );
        if depth >= self.path_len {
            // depth is within the heap-ordered subtree
            let d = depth - self.path_len;
            let start = self.path_len + ((1 << d) - 1);
            let end = self.path_len + ((1 << (d + 1)) - 1);
            (start, end)
        } else {
            // vertical path layer: single node per level
            (depth, depth + 1)
        }
    }
}

/// A Memory Merkle tree composed of independent subtrees (one per address space),
/// each built asynchronously and finalized into a top-level Merkle root.
///
/// Layout:
/// - The memory is split across multiple `MemoryMerkleSubTree` instances, one per address space.
/// - The top-level tree is formed by hashing all subtree roots into a single buffer (`top_roots`).
///     - top_roots layout: \[root, hash(root_addr_space_1, root_addr_space_2),
///       hash(root_addr_space_3), hash(root_addr_space_4), ...\]
///     - if we have > 4 address spaces, top_roots will be extended with the next hash, etc.
///
/// Execution:
/// - Subtrees are built asynchronously on individual HIP streams.
/// - The final root is computed after all subtrees complete, on a shared stream.
/// - `HipEvent`s are used to synchronize subtree completion.
pub struct MemoryMerkleTree {
    pub stream: Arc<HipStream>,
    pub subtrees: Vec<MemoryMerkleSubTree>,
    pub top_roots: DeviceBuffer<H>,
    zero_hash: DeviceBuffer<H>,
    pub height: usize,
    pub hasher_buffer: SharedBuffer<F>,
    mem_config: MemoryConfig,
}

impl MemoryMerkleTree {
    /// Creates a full Merkle tree with one subtree per address space.
    /// Initializes all buffers and precomputes the zero hash chain.
    pub fn new(mem_config: MemoryConfig, hasher_chip: Arc<Poseidon2PeripheryChipHIP>) -> Self {
        let addr_space_sizes = mem_config
            .addr_spaces
            .iter()
            .map(|ashc| {
                assert!(
                    ashc.num_cells % DIGEST_WIDTH == 0,
                    "the number of cells must be divisible by `DIGEST_WIDTH`"
                );
                ashc.num_cells / DIGEST_WIDTH
            })
            .collect::<Vec<_>>();
        assert!(!(addr_space_sizes.is_empty()), "Invalid config");

        let num_addr_spaces = addr_space_sizes.len() - ADDR_SPACE_OFFSET as usize;
        assert!(
            num_addr_spaces.is_power_of_two(),
            "Number of address spaces must be a one plus power of two"
        );
        for &sz in addr_space_sizes.iter().take(ADDR_SPACE_OFFSET as usize) {
            assert!(
                sz == 0,
                "The first `ADDR_SPACE_OFFSET` address spaces are assumed to be empty"
            );
        }

        let label_max_bits = mem_config.pointer_max_bits - log2_ceil_usize(DIGEST_WIDTH);

        let zero_hash = DeviceBuffer::<H>::with_capacity(label_max_bits + 1);
        let top_roots = DeviceBuffer::<H>::with_capacity(2 * num_addr_spaces - 1);
        unsafe {
            calculate_zero_hash(&zero_hash, label_max_bits).unwrap();
        }

        Self {
            stream: Arc::new(HipStream::new().unwrap()),
            subtrees: Vec::new(),
            top_roots,
            height: label_max_bits + log2_ceil_usize(num_addr_spaces),
            zero_hash,
            hasher_buffer: hasher_chip.shared_buffer(),
            mem_config,
        }
    }

    pub fn mem_config(&self) -> &MemoryConfig {
        &self.mem_config
    }

    /// Starts asynchronous construction of the specified address space's Merkle subtree.
    /// Uses internal zero hashes and launches kernels on the subtree's own HIP stream.
    ///
    /// Here `addr_space` is the _unshifted_ address space, so `addr_space = 0` is the immediate
    /// address space, which should be ignored.
    pub fn build_async(&mut self, d_data: &DeviceBuffer<u8>, addr_space: usize) {
        if addr_space < ADDR_SPACE_OFFSET as usize {
            return;
        }
        let addr_space_idx = addr_space - ADDR_SPACE_OFFSET as usize;
        if addr_space < self.mem_config.addr_spaces.len() && addr_space_idx == self.subtrees.len() {
            let mut subtree = MemoryMerkleSubTree::new(
                self.mem_config.addr_spaces[addr_space].num_cells / DIGEST_WIDTH,
                1 << (self.zero_hash.len() - 1), /* label_max_bits */
            );
            subtree.build_async(d_data, addr_space_idx, &self.zero_hash);
            self.subtrees.push(subtree);
        } else {
            panic!("Invalid address space index");
        }
    }

    /// Finalizes the Merkle tree by collecting all subtree roots and computing the final root.
    /// Waits for all subtrees to complete and then performs the final hash operation.
    pub fn finalize(&self) {
        for subtree in self.subtrees.iter() {
            self.stream.wait(subtree.event.as_ref().unwrap()).unwrap();
        }

        let we_can_gather_bufs_event = HipEvent::new().unwrap();
        unsafe {
            we_can_gather_bufs_event
                .record(self.stream.as_raw())
                .unwrap();
        }
        default_stream_wait(&we_can_gather_bufs_event).unwrap();

        let roots: Vec<usize> = self
            .subtrees
            .iter()
            .map(|subtree| subtree.buf.as_ptr() as usize)
            .collect();
        let d_roots = roots.to_device().unwrap();
        let to_device_event = HipEvent::new().unwrap();
        unsafe {
            to_device_event.record(hipStreamPerThread).unwrap();
        }
        self.stream.wait(&to_device_event).unwrap();

        unsafe {
            finalize_merkle_tree(
                &d_roots,
                &self.top_roots,
                self.subtrees.len(),
                self.stream.as_raw(),
            )
            .unwrap();
        }

        self.stream.synchronize().unwrap();
    }

    /// Drops all massive buffers to free memory. Used at the end of an execution segment.
    pub fn drop_subtrees(&mut self) {
        self.subtrees = Vec::new();
    }

    /// Updates the tree and returns the merkle trace.
    pub fn update_with_touched_blocks(
        &self,
        unpadded_height: usize,
        d_touched_blocks: &DeviceBuffer<u32>, // consists of (as, label, ts, [F; 8])
        empty_touched_blocks: bool,
    ) -> AirProvingContext<HipBackend> {
        let mut public_values = self.top_roots.to_host().unwrap()[0].to_vec();
        let merkle_trace = {
            let width = MemoryMerkleCols::<u8, DIGEST_WIDTH>::width();
            let padded_height = next_power_of_two_or_zero(unpadded_height);
            let output = DeviceMatrix::<F>::with_capacity(padded_height, width);
            output.buffer().fill_zero().unwrap();

            let actual_heights = self.subtrees.iter().map(|s| s.height).collect::<Vec<_>>();
            let subtrees_pointers = self
                .subtrees
                .iter()
                .map(|st| st.buf.as_ptr() as usize)
                .collect::<Vec<_>>()
                .to_device()
                .unwrap();
            unsafe {
                update_merkle_tree(
                    &output,
                    &subtrees_pointers,
                    &self.top_roots,
                    &self.zero_hash,
                    d_touched_blocks,
                    self.height - log2_ceil_usize(self.subtrees.len()),
                    &actual_heights,
                    unpadded_height,
                    &self.hasher_buffer,
                )
                .unwrap();
            }

            if empty_touched_blocks {
                // The trace is small then
                let mut output_vec = output.to_host().unwrap();
                output_vec[unpadded_height - 1 + (width - 2) * padded_height] = F::ONE; // left_direction_different
                output_vec[unpadded_height - 1 + (width - 1) * padded_height] = F::ONE; // right_direction_different
                DeviceMatrix::new(
                    Arc::new(output_vec.to_device().unwrap()),
                    padded_height,
                    width,
                )
            } else {
                output
            }
        };
        public_values.extend(self.top_roots.to_host().unwrap()[0].to_vec());

        AirProvingContext::new(Vec::new(), Some(merkle_trace), public_values)
    }

    /// An auxiliary function to calculate the required number of rows for the merkle trace.
    pub fn calculate_unpadded_height(
        &self,
        touched_memory: &TimestampedEquipartition<F, DIGEST_WIDTH>,
    ) -> usize {
        let md = self.mem_config.memory_dimensions();
        let tree_height = md.overall_height();
        let shift_address = |(sp, ptr): (u32, u32)| (sp, ptr / DIGEST_WIDTH as u32);
        2 * if touched_memory.is_empty() {
            tree_height
        } else {
            tree_height
                + (0..(touched_memory.len() - 1))
                    .into_par_iter()
                    .map(|i| {
                        let x = md.label_to_index(shift_address(touched_memory[i].0));
                        let y = md.label_to_index(shift_address(touched_memory[i + 1].0));
                        (x ^ y).ilog2() as usize
                    })
                    .sum::<usize>()
        }
    }
}

impl Drop for MemoryMerkleTree {
    fn drop(&mut self) {
        // Force synchronize all streams in merkle tree before dropping.
        for s in &self.subtrees {
            s.stream.synchronize().unwrap();
        }
        self.stream.synchronize().unwrap();
        self.drop_subtrees();
    }
}
