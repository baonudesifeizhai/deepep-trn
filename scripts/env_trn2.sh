# Source inside the sgl-plugin-dev container before running anything on device.
[ -f /opt/torch-neuronx/.venv/bin/activate ] && source /opt/torch-neuronx/.venv/bin/activate
export OMP_NUM_THREADS=1 NEURON_RT_VISIBLE_CORES=0-3 NEURON_RT_EXEC_TIMEOUT=60
export NKI_ENABLE_TRACE_CACHE=0 NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export TORCH_NEURONX_NEFF_CACHE_DIR=${TORCH_NEURONX_NEFF_CACHE_DIR:-/tmp/deep_ep_trn_neff_cache}
