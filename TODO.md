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

https://prod.liveshare.vsengsaas.visualstudio.com/join?293B8EA27EEC725D15C109FBE8330030F3FD

## Meeting 15.11

3D Conv UNet in a SurfaceNet Fashion

Similar to VisionTransformer:
3D voxel patch with feature/RGB values -> latent1 -> 3D voxel patch with feature/RGB values 
Occ -> latent2 -> Occ
latent1 -> latent2 through transformer

## ToDO bis morgen:

[] SurfaceNet Speicherproblem beseitigen 
  [] Speicher aufräumen (load->delete->save) (Luis)
  [] Projection hinzufügen zu VoxelGridTransform (Luis)
[] Mast3r Baseline
  [] Führe Mast3r und speicher Resulutate (Luca)
  [] Mast3r Transform -> lade GT (Luca)
  [] einmal global alignment callen für scenes 
  [] einmal transform based on camera extrinsics
  [] PointCloud -> VoxelGrid
[] neue Pipeline 

6.12. Meeting:
[] Prepare faster iteration times and grid search with termination criteria
[]


# Data sampling

# Split Loading
- [ ] Add Dataset Type

## Chunk selection
- [x] Relative Camera Pose (Angle)
- [x] Area under K
- [ ] Distance in trajectory
- [ ] Larger chunk (with sampling in transform)

def heuristic(seminal_img, images, occ) -> Float[#images]
  pass

### Rejected

- [ ] Amount of geometry (#occupied) (with target)

## Pair selection

- [ ] Combinatorik
- [ ] Only to seminal image

## Transform

- [ ] 3d Point based (weighting mask)
- [ ] Smearing based

## Model

- [ ] Basic (U-Net)
- [ ] Transformer Basic (U-Net)

- [ ] Working model

## Performance

- [ ] BFloat
- [ ] Pages
- [ ] 


# Dataloading to implement again
- [ ] Shuffling (deleted)
- [ ] Positional Encoding as a Transform
- [ ] Seperate image pairs
- [ ] Normalization as Transform