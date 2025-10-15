import numpy as np

# Strain and stress data
eps_1_xx, eps_1_xy, eps_1_yx, eps_1_yy = 1.0000806934780572e-08,2.982246552658268e-13,-1.246575254231623e-12,2.4792492780621555e-12
sig_1_xx, sig_1_xy, sig_1_yx, sig_1_yy = 8.227020094496678,-2.0229212228672,-2.025624149769488,-3.693554592606602
eps_2_xx, eps_2_xy, eps_2_yx, eps_2_yy = -1.3286222947449637e-13,3.0012440923066122e-12,-3.6811262516943554e-12,1.0000150211045088e-08
sig_2_xx, sig_2_xy, sig_2_yx, sig_2_yy = -3.682884878463236,5.20271562793989,5.2132279023914085,34.798993330211225
eps_3_xx, eps_3_xy, eps_3_yx, eps_3_yy = 1.144653344671966e-12,1.0002622018970677e-08,9.997812641295334e-09,1.3814895273385999e-12
sig_3_xx, sig_3_xy, sig_3_yx, sig_3_yy = -4.102156566683894,47.39957975183041,47.4009450879623,10.401935938635742
# eps_1_xx, eps_1_xy, eps_1_yx, eps_1_yy = 1.0000261037311868e-08, -1.3613807032060596e-12, -1.880233218439928e-12, 4.34507216637931e-12
# sig_1_xx, sig_1_xy, sig_1_yx, sig_1_yy = 497.56619963559183, -0.4924775889589401, -0.495217577504508, 102.7704560347958
# eps_2_xx, eps_2_xy, eps_2_yx, eps_2_yy = -8.034535947482412e-13, -2.576097035857024e-13, 1.0801610978271461e-12, 1.0001003409543796e-08
# sig_2_xx, sig_2_xy, sig_2_yx, sig_2_yy = 102.46939465799204, 0.4234565941704577, 0.4261966149406083, 497.4583834708085
# eps_3_xx, eps_3_xy, eps_3_yx, eps_3_yy = -2.4434575083951133e-12, 9.99947633603473e-09, 1.0000484958115342e-08, 2.868040676585486e-12
# sig_3_xx, sig_3_xy, sig_3_yx, sig_3_yy = -0.8885936123664435, 383.1570795168701, 383.162120089974, 0.8992333689480838

# Strain vectors (Voigt notation: [xx, yy, 2xy] or [xx, yy, 2yx])
# xy version
E_xy = np.array([
    [eps_1_xx, eps_1_yy, np.sqrt(2) * eps_1_xy],
    [eps_2_xx, eps_2_yy, np.sqrt(2) * eps_2_xy],
    [eps_3_xx, eps_3_yy, np.sqrt(2) * eps_3_xy]
]).T  # Transpose to get 3x3 with columns as strain vectors

# yx version
E_yx = np.array([
    [eps_1_xx, eps_1_yy, np.sqrt(2) * eps_1_yx],
    [eps_2_xx, eps_2_yy, np.sqrt(2) * eps_2_yx],
    [eps_3_xx, eps_3_yy, np.sqrt(2) * eps_3_yx]
]).T

# Stress vectors (Voigt notation: [xx, yy, xy])
S = np.array([
    [sig_1_xx, sig_1_yy, sig_1_xy],
    [sig_2_xx, sig_2_yy, sig_2_xy],
    [sig_3_xx, sig_3_yy, sig_3_xy]
]).T

# Compute C matrices: C = S * E^-1
C_xy = np.dot(S, np.linalg.inv(E_xy))
C_yx = np.dot(S, np.linalg.inv(E_yx))

# Print results
print("C matrix (xy strain version):")
print(C_xy)
print("\nC matrix (yx strain version):")
print(C_yx)

# import numpy as np

# # Given data
# # Strains [eps_xx, eps_yy, eps_xy]
# eps_1 = np.array([9.999664584520949e-09, 3.115677083264572e-13, -3.56e-13])
# eps_2 = np.array([-1.51e-13, 1.0000363509990722e-08, 7.660573407807254e-13])
# eps_3 = np.array([5.057999379076549e-13, -1.41e-13, 1.0000063706809322e-08])

# # Stresses [sig_xx, sig_yy, sig_xy]
# sig_1 = np.array([497.5837277818581, 102.44061658230174, 0.6867240290071829])
# sig_2 = np.array([102.41748358247132, 497.62228112841854, -0.672829084])
# sig_3 = np.array([1.4274048461578832, -1.380975297, 381.3590109669917])

# # Construct the strain matrix E (9 rows, 6 columns for C_1111, C_1122, C_1112, C_2222, C_1222, C_1212)
# E = np.zeros((9, 6))
# # State 1
# E[0, :] = [eps_1[0], eps_1[1], eps_1[2], 0, 0, 0]         # sig_xx
# E[1, :] = [0, eps_1[0], 0, eps_1[1], eps_1[2], 0]         # sig_yy
# E[2, :] = [0, 0, eps_1[0], 0, eps_1[1], eps_1[2]]         # sig_xy
# # State 2
# E[3, :] = [eps_2[0], eps_2[1], eps_2[2], 0, 0, 0]         # sig_xx
# E[4, :] = [0, eps_2[0], 0, eps_2[1], eps_2[2], 0]         # sig_yy
# E[5, :] = [0, 0, eps_2[0], 0, eps_2[1], eps_2[2]]         # sig_xy
# # State 3
# E[6, :] = [eps_3[0], eps_3[1], eps_3[2], 0, 0, 0]         # sig_xx
# E[7, :] = [0, eps_3[0], 0, eps_3[1], eps_3[2], 0]         # sig_yy
# E[8, :] = [0, 0, eps_3[0], 0, eps_3[1], eps_3[2]]         # sig_xy

# # Stress vector S (9 rows)
# S = np.concatenate([sig_1, sig_2, sig_3])

# # Solve for C using least squares (E @ C = S)
# C_flat, residuals, rank, s = np.linalg.lstsq(E, S, rcond=None)
# C_1111, C_1122, C_1112, C_2222, C_1222, C_1212 = C_flat

# # Construct the 3x3 C matrix
# C = np.array([
#     [C_1111, C_1122, C_1112],
#     [C_1122, C_2222, C_1222],
#     [C_1112, C_1222, C_1212]
# ])

# print("Derived Stiffness Matrix C:")
# print(C)

# # Verify by recalculating stresses
# sig_1_calc = C @ eps_1
# sig_2_calc = C @ eps_2
# sig_3_calc = C @ eps_3

# print("\nVerification:")
# print("State 1 - Calculated:", sig_1_calc, "Given:", sig_1)
# print("State 2 - Calculated:", sig_2_calc, "Given:", sig_2)
# print("State 3 - Calculated:", sig_3_calc, "Given:", sig_3)