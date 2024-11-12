# PROJECT TODOs

## Data Processing
- [ ] Masking of bs data (using iphone depth)
- [x] Chunking and sampling of distance values (possibly visibility based) -> store (point + value (GT))
  - smart way to sample
  
- [ ] Run preprocessing on X% of the data
- [ ] Preparation of paired images
   - selection of image base set
   - Maybe based on some heuristic (eg. minimum angle)

## Mast3r Baseline 
- [x] Paired images -> stored output (X, C, F, latent feature (ViT Encoder), Before Head)
- [ ] Naive occupancy / DF creation for baseline metrics

## VoxL3R
- [ ] "Smeared" un-projection 
- [ ] pose known -> Rigid body
- [ ] architecture of fusion network
- [ ] Loss implementation
- [ ] Output Network + Sampling logic
- [ ] Training loop


## Comparison to 

https://prod.liveshare.vsengsaas.visualstudio.com/join?6AACC20CB3792E1E3E233447D9E3FE0494AC