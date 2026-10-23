"""Native Wan 2.2 I2V-A14B model definition.

A14B is a Wan2.1-shaped recipe with a Wan2.2-shaped model. The conditioning is
the 2.1 one -- every condition stream carries a four-channel temporal
first-frame mask ahead of its latent body, and the timestep stays a scalar --
so the task layer reuses the 2.1 encoder and cache contract rather than the
TI2V-5B in-painting one.

What is new is the model: two full 14B transformers, a high-noise expert and a
low-noise one, routed by where the sampled timestep falls against
``boundary_ratio``. Each expert is its own FSDP execution root and loads from
its own published subfolder.
"""
