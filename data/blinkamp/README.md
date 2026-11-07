# Vendored BlinkAMP motion assets

These 17 motion files, joint mapping and accompanying PhysHSI license were copied into this repository so BlinkAMP
does not need a sibling PhysHSI checkout at runtime. Raw tensors are unmodified; preprocessing is in
`src/blinkamp/motion.py`. Source paths, revision, frame rate and SHA-256 are in `configs/blinkamp/motions.json`.

Source: PhysHSI, https://github.com/InternRobotics/PhysHSI, imported from the user-provided reference checkout.
PhysHSI declares CC BY-NC-SA 4.0; its license is preserved in `PHYSHSI_LICENSE.txt`.
Retain underlying SAMP/AMASS/100STYLE data terms and source attribution where applicable.
These third-party motion assets are not relicensed under BlinkMimic's MIT code license.

Files are deliberately tracked despite general `/data` and `*.pt` ignore rules. Trained policies, generated
trajectories, the original BlinkMimic dataset and generated robot assets remain ignored.
Use `tools/import_amp_data.py` only when intentionally refreshing this copy and its manifest.
