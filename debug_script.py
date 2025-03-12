from einops import rearrange
import torch

num_pairs = 4
b = 16

x = y = z = torch.arange(32)
X, Y, Z = torch.meshgrid(x,y,z, indexing= 'ij')
grid = torch.stack([X,Y,Z], axis=0)
grid = grid.repeat(b, num_pairs, 1, 1, 1, 1) # B, P, C, X, Y, Z
grid_reshaped = rearrange(grid, 'b p c (X px) (Y py) (Z pz) -> b p (X Y Z) (px py pz c)', px = 2, py =2, pz = 2)
gird_reshaped_final = rearrange(grid_reshaped, "b p XYZ d -> b (XYZ p) d")

grid_back = rearrange(gird_reshaped_final, 'b (X Y Z p) (px py pz c) -> b p c (X px) (Y py) (Z pz)', p=num_pairs, X=16, Y=16, Z=16, px = 2, py = 2, pz = 2)
grid_back_2 = rearrange(grid_back, "b p c x y z -> (b p) c x y z")
y = rearrange(grid_back_2, "(b p) c x y z -> b p c x y z", b=b, p=num_pairs)

print(gird_reshaped_final[3, 0, :] ==  gird_reshaped_final[3, 3, :] )

idx = 4*144

q_idx = idx // num_pairs
grid_edge = 16
q_idx_x, q_idx_y, q_idx_z = q_idx // (grid_edge * grid_edge), (q_idx // grid_edge) % grid_edge, q_idx % grid_edge

print(q_idx_x, q_idx_y, q_idx_z)
print(gird_reshaped_final[3, idx, :])